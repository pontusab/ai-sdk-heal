import type {
  AssistantContent,
  ModelMessage,
  ToolCallPart,
  ToolContent,
  ToolResultPart,
} from "ai";
import type { Repair, Rule } from "../types";
import {
  TOOL_NAME_RE,
  assistantParts,
  hasSubstantiveContent,
  isAssistant,
  isReasoningPart,
  isTool,
  isToolCallPart,
  isToolResultPart,
  replaceAssistantContent,
  replaceToolContent,
  sanitizeToolName,
} from "../utils";

/**
 * Heal invalid tool names.
 *
 * Models (especially Claude) occasionally hallucinate tool names that include
 * XML fragments, newlines, quotes, or other junk (see vercel/ai#9141). Anthropic
 * and Bedrock both require ^[a-zA-Z0-9_-]{1,64}$ — anything else is rejected.
 *
 * We fix up both halves of the call/result pair so they stay linked.
 */
export const healInvalidToolName: Rule = (messages, { policy }) => {
  if (policy.invalidToolName === "keep")
    return { messages, repairs: [] };

  const repairs: Repair[] = [];
  const renames = new Map<string, string>();

  for (let mi = 0; mi < messages.length; mi++) {
    const m = messages[mi];
    if (!m) continue;
    if (!isAssistant(m)) continue;
    if (typeof m.content === "string") continue;
    for (let pi = 0; pi < m.content.length; pi++) {
      const p = m.content[pi];
      if (!isToolCallPart(p)) continue;
      if (TOOL_NAME_RE.test(p.toolName)) continue;
      const clean = sanitizeToolName(p.toolName);
      renames.set(p.toolCallId, clean);
      repairs.push({
        rule: "invalid-tool-name",
        messageIndex: mi,
        partIndex: pi,
        action: policy.invalidToolName === "drop-pair" ? "dropped-part" : "renamed",
        reason: `Tool name "${p.toolName}" violates ^[a-zA-Z0-9_-]{1,64}$`,
        toolCallId: p.toolCallId,
      });
    }
  }

  if (renames.size === 0) return { messages, repairs: [] };

  if (policy.invalidToolName === "drop-pair") {
    const drop = new Set(renames.keys());
    return {
      messages: dropToolCallsAndResults(messages, drop),
      repairs,
    };
  }

  const next = messages.map((m) => {
    if (isAssistant(m) && typeof m.content !== "string") {
      return replaceAssistantContent(
        m,
        m.content.map((p) => {
          if (isToolCallPart(p) && renames.has(p.toolCallId)) {
            const clean = renames.get(p.toolCallId)!;
            if (p.toolName === clean) return p;
            return { ...p, toolName: clean };
          }
          return p;
        }),
      );
    }
    if (isTool(m)) {
      return replaceToolContent(
        m,
        m.content.map((p) => {
          if (isToolResultPart(p) && renames.has(p.toolCallId)) {
            const clean = renames.get(p.toolCallId)!;
            if (p.toolName === clean) return p;
            return { ...p, toolName: clean };
          }
          return p;
        }),
      );
    }
    return m;
  });

  return { messages: next, repairs };
};

/**
 * Heal invalid tool-call inputs (vercel/ai#13645).
 *
 * When the model emits malformed JSON for tool inputs, AI SDK stores the raw
 * string in `rawInput` and `convertToModelMessages` can serialize it as the
 * tool_use.input. Anthropic requires a dictionary, so the conversation is
 * bricked forever.
 *
 * We coerce any non-object input to `{ raw: "<string>" }` (or `{}` under
 * `empty-object` policy) so the conversation stays usable.
 */
export const healInvalidToolInput: Rule = (messages, { policy }) => {
  if (policy.invalidToolInput === "keep")
    return { messages, repairs: [] };

  const repairs: Repair[] = [];
  const next = messages.map((m, mi) => {
    if (!isAssistant(m) || typeof m.content === "string") return m;
    let changed = false;
    const content = m.content.map((p, pi) => {
      if (!isToolCallPart(p)) return p;
      if (isValidToolInput(p.input)) return p;
      changed = true;
      const replacement =
        policy.invalidToolInput === "empty-object"
          ? {}
          : coerceInputToObject(p.input);
      repairs.push({
        rule: "invalid-tool-input",
        messageIndex: mi,
        partIndex: pi,
        action: "coerced-input",
        reason: `Tool input was ${typeof p.input}; coerced to object`,
        toolCallId: p.toolCallId,
      });
      return { ...p, input: replacement };
    });
    return changed ? replaceAssistantContent(m, content) : m;
  });

  return { messages: next, repairs };
};

function isValidToolInput(x: unknown): boolean {
  return (
    x !== null &&
    typeof x === "object" &&
    !Array.isArray(x) &&
    !(x instanceof Date)
  );
}

function coerceInputToObject(x: unknown): Record<string, unknown> {
  if (x === undefined || x === null) return {};
  if (typeof x === "string") {
    const trimmed = x.trim();
    if (trimmed.length === 0) return {};
    try {
      const parsed = JSON.parse(trimmed);
      if (isValidToolInput(parsed)) return parsed as Record<string, unknown>;
    } catch {
      // ignore
    }
    return { raw: x };
  }
  if (Array.isArray(x)) return { values: x };
  return { value: x };
}

/**
 * Drop duplicate tool-results (same toolCallId appearing twice in the history).
 * This happens when a retry replays an old turn alongside the new result.
 */
export const healDuplicateToolResult: Rule = (messages, { policy }) => {
  if (policy.duplicateToolResult === "keep")
    return { messages, repairs: [] };

  const repairs: Repair[] = [];
  const keepLast = policy.duplicateToolResult === "dedupe-last";
  const seen = new Map<string, { mi: number; pi: number }>();

  // Walk in desired order; remember the one to keep.
  const iter = keepLast
    ? rangeReversed(messages.length)
    : range(messages.length);
  for (const mi of iter) {
    const m = messages[mi];
    if (!m || !isTool(m)) continue;
    m.content.forEach((p, pi) => {
      if (!isToolResultPart(p)) return;
      if (seen.has(p.toolCallId)) {
        repairs.push({
          rule: "duplicate-tool-result",
          messageIndex: mi,
          partIndex: pi,
          action: "dropped-part",
          reason: `Duplicate tool-result for id ${p.toolCallId}`,
          toolCallId: p.toolCallId,
        });
      } else {
        seen.set(p.toolCallId, { mi, pi });
      }
    });
  }

  if (repairs.length === 0) return { messages, repairs: [] };

  const dropKeyOf = (mi: number, pi: number) => `${mi}:${pi}`;
  const dropKeys = new Set(
    repairs.map((r) => dropKeyOf(r.messageIndex, r.partIndex ?? -1)),
  );

  const next = messages.map((m, mi) => {
    if (!isTool(m)) return m;
    const content = m.content.filter(
      (_, pi) => !dropKeys.has(dropKeyOf(mi, pi)),
    );
    return content.length === m.content.length
      ? m
      : replaceToolContent(m, content);
  });

  return { messages: next, repairs };
};

/**
 * Pair tool calls with their results.
 *
 * Invariants we enforce:
 *   - Every assistant `tool-call` that is NOT provider-executed must have a
 *     matching `tool-result` in the next tool message (vercel/ai#8516, #14259).
 *   - Every `tool-result` must reference an assistant `tool-call` that
 *     precedes it.
 *
 * Policies:
 *   - `stub-result` (default): insert a placeholder tool-result with
 *     `type: "error-text"` explaining the call was aborted. This lets the
 *     model see the failure and recover.
 *   - `drop-call`: drop the dangling tool-call so the history is well-formed
 *     without inventing data.
 */
export const healToolPairing: Rule = (messages, { policy }) => {
  const repairs: Repair[] = [];

  type CallRef = { mi: number; pi: number; part: ToolCallPart };
  const calls = new Map<string, CallRef>();

  for (let mi = 0; mi < messages.length; mi++) {
    const m = messages[mi];
    if (!m) continue;
    if (isAssistant(m) && typeof m.content !== "string") {
      m.content.forEach((p, pi) => {
        if (isToolCallPart(p) && !p.providerExecuted) {
          calls.set(p.toolCallId, { mi, pi, part: p });
        }
      });
    }
  }

  const resolved = new Set<string>();
  const unknownResults: Array<{ mi: number; pi: number; id: string }> = [];

  for (let mi = 0; mi < messages.length; mi++) {
    const m = messages[mi];
    if (!m || !isTool(m)) continue;
    m.content.forEach((p, pi) => {
      if (!isToolResultPart(p)) return;
      if (calls.has(p.toolCallId)) {
        resolved.add(p.toolCallId);
      } else {
        unknownResults.push({ mi, pi, id: p.toolCallId });
      }
    });
  }

  const unresolved = [...calls.values()].filter(
    (c) => !resolved.has(c.part.toolCallId),
  );

  let next = messages;

  if (unresolved.length > 0) {
    if (policy.orphanToolUse === "drop-call") {
      const dropIds = new Set(unresolved.map((c) => c.part.toolCallId));
      next = dropToolCallsAndResults(next, dropIds);
      for (const c of unresolved) {
        repairs.push({
          rule: "orphan-tool-use",
          messageIndex: c.mi,
          partIndex: c.pi,
          action: "dropped-part",
          reason: `Tool call ${c.part.toolName} (${c.part.toolCallId}) had no matching result`,
          toolCallId: c.part.toolCallId,
        });
      }
    } else if (policy.orphanToolUse === "stub-result") {
      next = insertStubResults(next, unresolved, repairs);
    } else {
      for (const c of unresolved) {
        repairs.push({
          rule: "orphan-tool-use",
          messageIndex: c.mi,
          partIndex: c.pi,
          action: "dropped-part",
          reason: `Tool call ${c.part.toolName} (${c.part.toolCallId}) had no matching result (policy: keep)`,
          toolCallId: c.part.toolCallId,
        });
      }
    }
  }

  if (unknownResults.length > 0 && policy.orphanToolResult === "drop") {
    const dropKeyOf = (mi: number, pi: number) => `${mi}:${pi}`;
    const dropKeys = new Set(
      unknownResults.map((r) => dropKeyOf(r.mi, r.pi)),
    );
    // Remap indexes after orphan-tool-use edits may have moved them.
    // Simplest correct approach: operate on the current `next` by filtering
    // tool messages whose results reference unknown ids.
    const unknownIds = new Set(unknownResults.map((r) => r.id));
    next = next.map((m, mi) => {
      if (!isTool(m)) return m;
      const filtered = m.content.filter((p) => {
        if (!isToolResultPart(p)) return true;
        return !unknownIds.has(p.toolCallId);
      });
      if (filtered.length === m.content.length) return m;
      return replaceToolContent(m, filtered);
    });
    for (const r of unknownResults) {
      repairs.push({
        rule: "orphan-tool-result",
        messageIndex: r.mi,
        partIndex: r.pi,
        action: "dropped-part",
        reason: `Tool result references unknown tool call id ${r.id}`,
        toolCallId: r.id,
      });
    }
    void dropKeys;
  }

  return { messages: next, repairs };
};

/**
 * Drop assistant messages that are empty or contain only whitespace. The
 * provider will reject them on submission.
 */
export const healEmptyAssistantMessage: Rule = (messages, { policy }) => {
  if (policy.emptyAssistantMessage === "keep")
    return { messages, repairs: [] };

  const repairs: Repair[] = [];
  const next: ModelMessage[] = [];
  for (let mi = 0; mi < messages.length; mi++) {
    const m = messages[mi];
    if (!m) continue;
    if (isAssistant(m) && !hasSubstantiveContent(m.content)) {
      // Don't drop if it contains reasoning — that's handled by the
      // anthropic-specific rule which knows whether the provider allows it.
      const parts = assistantParts(m);
      const onlyReasoning = parts.length > 0 && parts.every(isReasoningPart);
      if (!onlyReasoning) {
        repairs.push({
          rule: "empty-assistant-message",
          messageIndex: mi,
          action: "dropped-message",
          reason: "Assistant message has no substantive content",
        });
        continue;
      }
    }
    next.push(m);
  }
  return { messages: next, repairs };
};

// ─── helpers ────────────────────────────────────────────────────────────

function* range(n: number): Generator<number> {
  for (let i = 0; i < n; i++) yield i;
}

function* rangeReversed(n: number): Generator<number> {
  for (let i = n - 1; i >= 0; i--) yield i;
}

function dropToolCallsAndResults(
  messages: ModelMessage[],
  ids: Set<string>,
): ModelMessage[] {
  return messages.map((m) => {
    if (isAssistant(m) && typeof m.content !== "string") {
      const content = m.content.filter(
        (p) => !(isToolCallPart(p) && ids.has(p.toolCallId)),
      );
      if (content.length === m.content.length) return m;
      return replaceAssistantContent(m, content as AssistantContent);
    }
    if (isTool(m)) {
      const content = m.content.filter(
        (p) => !(isToolResultPart(p) && ids.has(p.toolCallId)),
      );
      if (content.length === m.content.length) return m;
      return replaceToolContent(m, content as ToolContent);
    }
    return m;
  });
}

/**
 * For each unresolved tool-call, insert a stub tool-result in the next tool
 * message after the call (or create one if none exists).
 */
function insertStubResults(
  messages: ModelMessage[],
  unresolved: Array<{ mi: number; pi: number; part: ToolCallPart }>,
  repairs: Repair[],
): ModelMessage[] {
  // Group unresolved calls by the message they appear in. For each group we
  // place stubs into the message immediately following — creating a tool
  // message if necessary.
  const byAssistantIndex = new Map<number, ToolCallPart[]>();
  for (const u of unresolved) {
    const list = byAssistantIndex.get(u.mi) ?? [];
    list.push(u.part);
    byAssistantIndex.set(u.mi, list);
    repairs.push({
      rule: "orphan-tool-use",
      messageIndex: u.mi,
      partIndex: u.pi,
      action: "inserted-part",
      reason: `Inserted stub tool-result for orphan call ${u.part.toolName} (${u.part.toolCallId})`,
      toolCallId: u.part.toolCallId,
    });
  }

  // Build the result bottom-up so insertion indices stay stable.
  const sortedKeys = [...byAssistantIndex.keys()].sort((a, b) => b - a);
  const out = [...messages];
  for (const idx of sortedKeys) {
    const parts = byAssistantIndex.get(idx)!;
    const stubs: ToolResultPart[] = parts.map((p) => ({
      type: "tool-result",
      toolCallId: p.toolCallId,
      toolName: p.toolName,
      output: {
        type: "error-text",
        value:
          "Tool call was not completed and no result was recorded. Assume the operation failed or was interrupted.",
      },
    }));
    const next = out[idx + 1];
    if (next && isTool(next)) {
      const existingIds = new Set(
        next.content
          .filter(isToolResultPart)
          .map((r) => r.toolCallId),
      );
      const newOnes = stubs.filter((s) => !existingIds.has(s.toolCallId));
      if (newOnes.length === 0) continue;
      out[idx + 1] = replaceToolContent(next, [...next.content, ...newOnes]);
    } else {
      out.splice(idx + 1, 0, {
        role: "tool",
        content: stubs,
      });
    }
  }
  return out;
}
