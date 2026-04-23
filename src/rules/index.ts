import type { Provider, Rule } from "../types";
import {
  healOrphanReasoningOnlyMessage,
  healMissingReasoningSignature,
} from "./anthropic";
import { healReasoningWithoutFollowingItem } from "./openai";
import {
  healDuplicateToolResult,
  healEmptyAssistantMessage,
  healInvalidToolInput,
  healInvalidToolName,
  healToolPairing,
} from "./shared";

export {
  healOrphanReasoningOnlyMessage,
  healMissingReasoningSignature,
  healReasoningWithoutFollowingItem,
  healDuplicateToolResult,
  healEmptyAssistantMessage,
  healInvalidToolInput,
  healInvalidToolName,
  healToolPairing,
};

/**
 * Rules run in a fixed order so later rules can assume earlier invariants:
 *
 *   1. invalid-tool-name   — rename before we match up pairs
 *   2. invalid-tool-input  — coerce before we match up pairs
 *   3. duplicate-tool-result — dedupe before pairing logic
 *   4. tool-pairing        — orphan tool_use / tool_result
 *   5. provider reasoning rules — work on the settled structure
 *   6. empty-assistant-message — final cleanup
 */
export function rulesFor(provider: Provider | undefined): Rule[] {
  const rules: Rule[] = [
    healInvalidToolName,
    healInvalidToolInput,
    healDuplicateToolResult,
    healToolPairing,
  ];

  if (provider === "anthropic" || provider === "bedrock-anthropic") {
    rules.push(healMissingReasoningSignature, healOrphanReasoningOnlyMessage);
  } else if (provider === "openai") {
    rules.push(healReasoningWithoutFollowingItem);
  } else if (provider === "google") {
    // Google tolerates reasoning blocks without signatures; no extra rules
    // beyond shared ones today.
  }

  rules.push(healEmptyAssistantMessage);
  return rules;
}
