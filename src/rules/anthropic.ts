import type { Repair, Rule } from "../types";
import {
  type ReasoningPart,
  isAssistant,
  isReasoningOnlyAssistant,
  isReasoningPart,
  replaceAssistantContent,
} from "../utils";

/**
 * Anthropic rejects assistant messages that contain only reasoning blocks
 * (see vercel/ai#13430). This happens after `pruneMessages` removes the
 * tool_call/tool_result pair but leaves the reasoning that preceded them.
 *
 * Fix: drop the whole assistant message. The reasoning without any paired
 * output is useless to the model anyway.
 */
export const healOrphanReasoningOnlyMessage: Rule = (messages, { policy }) => {
  if (policy.orphanReasoningOnlyMessage === "keep")
    return { messages, repairs: [] };

  const repairs: Repair[] = [];
  const next = messages.filter((m, mi) => {
    if (!isAssistant(m)) return true;
    if (!isReasoningOnlyAssistant(m)) return true;
    repairs.push({
      rule: "orphan-reasoning-only-message",
      messageIndex: mi,
      action: "dropped-message",
      reason:
        "Assistant message contains only reasoning parts; Anthropic rejects these",
    });
    return false;
  });
  return { messages: next, repairs };
};

/**
 * Anthropic extended thinking requires that any reasoning block we send back
 * to the API include its original `signature`. If we have reasoning without a
 * signature (vercel/ai#11602, #7729), sending it produces a 400.
 *
 * We drop unsigned reasoning by default. The alternative — shipping an
 * unsigned block — always fails.
 */
export const healMissingReasoningSignature: Rule = (messages, { policy }) => {
  if (policy.missingReasoningSignature === "keep")
    return { messages, repairs: [] };

  const repairs: Repair[] = [];
  const next = messages.map((m, mi) => {
    if (!isAssistant(m) || typeof m.content === "string") return m;
    let changed = false;
    const content = m.content.filter((p, pi) => {
      if (!isReasoningPart(p)) return true;
      if (hasAnthropicSignature(p)) return true;
      repairs.push({
        rule: "missing-reasoning-signature",
        messageIndex: mi,
        partIndex: pi,
        action: "dropped-part",
        reason:
          "Reasoning block has no anthropic.signature; Anthropic rejects unsigned thinking on replay",
      });
      changed = true;
      return false;
    });
    return changed ? replaceAssistantContent(m, content) : m;
  });
  return { messages: next, repairs };
};

function hasAnthropicSignature(p: ReasoningPart): boolean {
  // The ModelMessage shape uses `providerOptions`, but some persistence layers
  // store the `providerMetadata` emitted by stream events verbatim. Accept
  // either key to be forgiving of real-world data.
  const carrier =
    (p as { providerOptions?: unknown; providerMetadata?: unknown })
      .providerOptions ??
    (p as { providerMetadata?: unknown }).providerMetadata;
  if (!carrier || typeof carrier !== "object") return false;
  const anthropic = (carrier as Record<string, unknown>).anthropic;
  if (!anthropic || typeof anthropic !== "object") return false;
  const signature = (anthropic as Record<string, unknown>).signature;
  if (typeof signature === "string" && signature.length > 0) return true;
  const redactedData = (anthropic as Record<string, unknown>).redactedData;
  return typeof redactedData === "string" && redactedData.length > 0;
}
