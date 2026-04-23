import { describe, expect, test } from "bun:test";
import { healMessages } from "../heal";
import {
  duplicateToolResultFixture,
  invalidToolInputFixture,
  invalidToolNameFixture,
  orphanReasoningOnlyFixture,
  orphanToolResultFixture,
  orphanToolUseFixture,
  orphanToolUseOutputErrorFixture,
  unsignedReasoningFixture,
} from "./fixtures";
import { inferProvider } from "../heal";
import { isToolCallPart, isToolResultPart, TOOL_NAME_RE } from "../utils";
import type { AssistantModelMessage, ToolModelMessage } from "ai";

describe("healMessages — shared rules", () => {
  test("vercel/ai#8516: orphan tool_use → inserts stub result", () => {
    const { messages, repairs } = healMessages(orphanToolUseFixture(), {
      provider: "anthropic",
    });

    expect(repairs.some((r) => r.rule === "orphan-tool-use")).toBe(true);

    // The second assistant message's tool-call should now have a matching
    // tool-result in the following tool message.
    const callIds = collectToolCallIds(messages);
    const resultIds = collectToolResultIds(messages);
    for (const id of callIds) {
      expect(resultIds).toContain(id);
    }
  });

  test("vercel/ai#14259: orphan tool_use (output-error path) is resolved", () => {
    const { messages, repairs } = healMessages(
      orphanToolUseOutputErrorFixture(),
      { provider: "anthropic" },
    );
    expect(repairs.some((r) => r.rule === "orphan-tool-use")).toBe(true);
    const callIds = collectToolCallIds(messages);
    const resultIds = collectToolResultIds(messages);
    expect(callIds.length).toBeGreaterThan(0);
    for (const id of callIds) expect(resultIds).toContain(id);
  });

  test("vercel/ai#13645: string tool input is coerced to an object", () => {
    const { messages, repairs } = healMessages(invalidToolInputFixture(), {
      provider: "anthropic",
    });
    expect(repairs.some((r) => r.rule === "invalid-tool-input")).toBe(true);
    const assistant = messages.find(
      (m) => m.role === "assistant",
    ) as AssistantModelMessage;
    const call = (assistant.content as Array<unknown>).find(isToolCallPart);
    expect(call).toBeDefined();
    expect(typeof call!.input).toBe("object");
    expect(call!.input).not.toBe(null);
    expect(Array.isArray(call!.input)).toBe(false);
  });

  test("vercel/ai#9141: invalid tool name (XML fragment) is renamed, pair stays linked", () => {
    const { messages, repairs } = healMessages(invalidToolNameFixture(), {
      provider: "anthropic",
    });
    expect(repairs.some((r) => r.rule === "invalid-tool-name")).toBe(true);
    const assistant = messages.find(
      (m) => m.role === "assistant",
    ) as AssistantModelMessage;
    const tool = messages.find((m) => m.role === "tool") as ToolModelMessage;
    const call = (assistant.content as Array<unknown>).find(isToolCallPart)!;
    const result = tool.content.find(isToolResultPart)!;

    expect(TOOL_NAME_RE.test(call.toolName)).toBe(true);
    expect(call.toolName).toBe(result.toolName);
    expect(call.toolCallId).toBe(result.toolCallId);
  });

  test("duplicate tool-results are deduped (keep-last by default)", () => {
    const { messages, repairs } = healMessages(duplicateToolResultFixture());
    expect(repairs.some((r) => r.rule === "duplicate-tool-result")).toBe(true);
    const allResults = messages
      .filter((m) => m.role === "tool")
      .flatMap((m) => (m as ToolModelMessage).content.filter(isToolResultPart));
    const byId = new Map<string, number>();
    for (const r of allResults)
      byId.set(r.toolCallId, (byId.get(r.toolCallId) ?? 0) + 1);
    for (const count of byId.values()) expect(count).toBe(1);

    // And specifically the last one ("second (retried)") survives.
    const survived = allResults.find((r) => r.toolCallId === "dup");
    expect(survived).toBeDefined();
    expect(JSON.stringify(survived!.output)).toContain("second (retried)");
  });

  test("orphan tool-result is dropped", () => {
    const { messages, repairs } = healMessages(orphanToolResultFixture());
    expect(repairs.some((r) => r.rule === "orphan-tool-result")).toBe(true);
    const hasOrphan = messages.some(
      (m) =>
        m.role === "tool" &&
        (m as ToolModelMessage).content.some(
          (p) => isToolResultPart(p) && p.toolCallId === "nonexistent",
        ),
    );
    expect(hasOrphan).toBe(false);
  });
});

describe("healMessages — Anthropic rules", () => {
  test("vercel/ai#13430: reasoning-only assistant message is dropped", () => {
    const { messages, repairs } = healMessages(orphanReasoningOnlyFixture(), {
      provider: "anthropic",
    });
    expect(
      repairs.some((r) => r.rule === "orphan-reasoning-only-message"),
    ).toBe(true);
    expect(messages.length).toBe(2);
    expect(messages[0]!.role).toBe("user");
    expect(messages[1]!.role).toBe("user");
  });

  test("vercel/ai#11602: unsigned reasoning is dropped", () => {
    const { messages, repairs } = healMessages(unsignedReasoningFixture(), {
      provider: "anthropic",
    });
    expect(repairs.some((r) => r.rule === "missing-reasoning-signature")).toBe(
      true,
    );
    const assistant = messages.find(
      (m) => m.role === "assistant",
    ) as AssistantModelMessage;
    const hasReasoning = (assistant.content as Array<unknown>).some(
      (p) => (p as { type?: string }).type === "reasoning",
    );
    expect(hasReasoning).toBe(false);
    // But the text part survived.
    const hasText = (assistant.content as Array<unknown>).some(
      (p) => (p as { type?: string }).type === "text",
    );
    expect(hasText).toBe(true);
  });

  test("signed reasoning is preserved", () => {
    const signed = unsignedReasoningFixture();
    // Mutate to add a signature
    const assistant = signed[1] as AssistantModelMessage;
    (assistant.content as Array<unknown>)[0] = {
      type: "reasoning",
      text: "hmm the user wants...",
      providerOptions: { anthropic: { signature: "sig_abc" } },
    };
    const { repairs } = healMessages(signed, { provider: "anthropic" });
    expect(repairs.some((r) => r.rule === "missing-reasoning-signature")).toBe(
      false,
    );
  });
});

describe("healMessages — OpenAI rules", () => {
  test("trailing reasoning part is dropped", () => {
    const input = [
      { role: "user" as const, content: "hi" },
      {
        role: "assistant" as const,
        content: [
          { type: "text" as const, text: "ok" },
          { type: "reasoning" as const, text: "trailing thought" },
        ],
      },
    ];
    const { messages, repairs } = healMessages(input, { provider: "openai" });
    expect(
      repairs.some((r) => r.rule === "reasoning-without-following-item"),
    ).toBe(true);
    const assistant = messages[1] as AssistantModelMessage;
    expect((assistant.content as Array<unknown>).length).toBe(1);
  });
});

describe("healMessages — idempotence", () => {
  const cases = [
    ["orphanToolUse", orphanToolUseFixture()],
    ["orphanToolUseOutputError", orphanToolUseOutputErrorFixture()],
    ["invalidToolInput", invalidToolInputFixture()],
    ["invalidToolName", invalidToolNameFixture()],
    ["duplicateToolResult", duplicateToolResultFixture()],
    ["orphanReasoningOnly", orphanReasoningOnlyFixture()],
    ["unsignedReasoning", unsignedReasoningFixture()],
    ["orphanToolResult", orphanToolResultFixture()],
  ] as const;

  for (const [name, fixture] of cases) {
    test(`${name}: heal(heal(x)) === heal(x)`, () => {
      const first = healMessages(fixture, { provider: "anthropic" });
      const second = healMessages(first.messages, { provider: "anthropic" });
      expect(second.repairs).toEqual([]);
      expect(JSON.stringify(second.messages)).toBe(
        JSON.stringify(first.messages),
      );
    });
  }
});

describe("healMessages — valid input passthrough", () => {
  test("clean conversation is unchanged and reports no repairs", () => {
    const input = [
      { role: "user" as const, content: "hi" },
      {
        role: "assistant" as const,
        content: [
          {
            type: "tool-call" as const,
            toolCallId: "t1",
            toolName: "search",
            input: { q: "ok" },
          },
        ],
      },
      {
        role: "tool" as const,
        content: [
          {
            type: "tool-result" as const,
            toolCallId: "t1",
            toolName: "search",
            output: { type: "text" as const, value: "done" },
          },
        ],
      },
      {
        role: "assistant" as const,
        content: [{ type: "text" as const, text: "here you go" }],
      },
    ];
    const { messages, repairs } = healMessages(input, {
      provider: "anthropic",
    });
    expect(repairs).toEqual([]);
    expect(JSON.stringify(messages)).toBe(JSON.stringify(input));
  });
});

describe("inferProvider", () => {
  test("detects providers from AI SDK model.provider strings", () => {
    expect(inferProvider({ provider: "anthropic.messages" })).toBe("anthropic");
    expect(inferProvider({ provider: "openai.chat" })).toBe("openai");
    expect(inferProvider({ provider: "openai.responses" })).toBe("openai");
    expect(inferProvider({ provider: "google.generative-ai" })).toBe("google");
    expect(inferProvider({ provider: "google.vertex" })).toBe("google");
    expect(inferProvider({ provider: "amazon-bedrock.anthropic.messages" })).toBe(
      "bedrock-anthropic",
    );
    expect(inferProvider({ provider: "mock-provider" })).toBeUndefined();
    expect(inferProvider(undefined)).toBeUndefined();
    expect(inferProvider(null)).toBeUndefined();
    expect(inferProvider({})).toBeUndefined();
  });
});

describe("healMessages — options", () => {
  test("onRepair hook fires for each repair", () => {
    const fired: string[] = [];
    healMessages(orphanToolUseFixture(), {
      provider: "anthropic",
      onRepair: (r) => fired.push(r.rule),
    });
    expect(fired).toContain("orphan-tool-use");
  });

  test("throwOnRepair throws with MessageHealingError", () => {
    expect(() =>
      healMessages(orphanToolUseFixture(), {
        provider: "anthropic",
        throwOnRepair: true,
      }),
    ).toThrow(/ai-sdk-heal repaired/);
  });

  test("a throwing onRepair hook does not break healing", () => {
    const fired: string[] = [];
    const { repairs } = healMessages(orphanToolUseFixture(), {
      provider: "anthropic",
      onRepair: (r) => {
        fired.push(r.rule);
        throw new Error("observability hook exploded");
      },
    });
    expect(fired.length).toBeGreaterThan(0);
    expect(repairs.length).toBeGreaterThan(0);
  });

  test("policy: orphanToolUse=drop-call drops the call instead of stubbing", () => {
    const { messages } = healMessages(orphanToolUseFixture(), {
      provider: "anthropic",
      policy: { orphanToolUse: "drop-call" },
    });
    const callIds = collectToolCallIds(messages);
    expect(callIds).toEqual([]);
  });
});

// ─── helpers ────────────────────────────────────────────────────────────

function collectToolCallIds(
  messages: ReadonlyArray<{ role: string; content: unknown }>,
): string[] {
  const out: string[] = [];
  for (const m of messages) {
    if (m.role !== "assistant") continue;
    if (typeof m.content === "string") continue;
    for (const p of m.content as Array<unknown>) {
      if (isToolCallPart(p)) out.push(p.toolCallId);
    }
  }
  return out;
}

function collectToolResultIds(
  messages: ReadonlyArray<{ role: string; content: unknown }>,
): string[] {
  const out: string[] = [];
  for (const m of messages) {
    if (m.role !== "tool") continue;
    for (const p of m.content as Array<unknown>) {
      if (isToolResultPart(p)) out.push(p.toolCallId);
    }
  }
  return out;
}
