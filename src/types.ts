import type { ModelMessage } from "ai";

export type Provider = "anthropic" | "openai" | "google" | "bedrock-anthropic";

/** The rule that was applied to heal a message. */
export type RuleName =
  // Shared across providers
  | "orphan-tool-use"
  | "orphan-tool-result"
  | "invalid-tool-name"
  | "invalid-tool-input"
  | "empty-assistant-message"
  | "duplicate-tool-result"
  // Anthropic-specific
  | "orphan-reasoning-only-message"
  | "missing-reasoning-signature"
  // OpenAI-specific (Responses API)
  | "reasoning-without-following-item";

export interface Repair {
  rule: RuleName;
  /** index of the message in the input array, -1 if rule inserted a new message */
  messageIndex: number;
  /** index of the part inside the message, -1 if whole message */
  partIndex?: number;
  action:
    | "dropped-message"
    | "dropped-part"
    | "inserted-message"
    | "inserted-part"
    | "replaced-part"
    | "reordered-parts"
    | "renamed"
    | "coerced-input";
  /** Human-readable reason. */
  reason: string;
  /** tool call id if applicable. */
  toolCallId?: string;
}

/**
 * Policy controls what action the healer takes when it detects a broken state.
 * Defaults are chosen to be safe: keep conversation alive, stub missing data
 * with explicit placeholders, never silently lose user content.
 */
export interface Policy {
  /**
   * Assistant `tool-call` with no matching `tool-result` in the following tool message.
   * - `stub-result` (default): insert a placeholder tool-result saying the call was aborted
   * - `drop-call`: remove the dangling tool-call entirely
   * - `keep`: leave it (will usually produce a 400 from the provider)
   */
  orphanToolUse?: "stub-result" | "drop-call" | "keep";

  /**
   * `tool-result` without a matching prior `tool-call`.
   * - `drop` (default)
   * - `keep`
   */
  orphanToolResult?: "drop" | "keep";

  /**
   * Tool name contains invalid characters (see #9141 - models hallucinating XML fragments
   * as tool names). Anthropic + Bedrock require `^[a-zA-Z0-9_-]{1,64}$`.
   * - `rename` (default): slug the name, keep the call
   * - `drop-pair`: drop both tool-call and paired tool-result
   * - `keep`
   */
  invalidToolName?: "rename" | "drop-pair" | "keep";

  /**
   * Tool input was left as a raw string (see #13645 - model hallucinated invalid JSON,
   * AI SDK stored rawInput, Anthropic 400s forever).
   * - `coerce-object` (default): wrap as `{ raw: string }`
   * - `empty-object`: replace with `{}`
   * - `keep`
   */
  invalidToolInput?: "coerce-object" | "empty-object" | "keep";

  /**
   * Assistant message that is empty or has only whitespace text.
   * - `drop` (default)
   * - `keep`
   */
  emptyAssistantMessage?: "drop" | "keep";

  /**
   * Two tool-results with the same toolCallId (caused by bad retries/persistence).
   * - `dedupe-last` (default): keep the last one, drop earlier duplicates
   * - `dedupe-first`: keep the first
   * - `keep`
   */
  duplicateToolResult?: "dedupe-last" | "dedupe-first" | "keep";

  /**
   * Anthropic: assistant message left with only `reasoning` parts after pruning
   * (see #13430). Anthropic rejects reasoning-only messages.
   * - `drop-message` (default)
   * - `keep`
   */
  orphanReasoningOnlyMessage?: "drop-message" | "keep";

  /**
   * Anthropic extended thinking: reasoning blocks that never got a signature (see
   * #11602). Sending them back without signature hits a 400, but dropping loses context.
   * - `drop-reasoning` (default): drop the unsigned reasoning parts
   * - `keep`
   */
  missingReasoningSignature?: "drop-reasoning" | "keep";

  /**
   * OpenAI responses API: `reasoning` item must be followed by a same-flow item
   * (see #8379).
   * - `drop-reasoning` (default)
   * - `keep`
   */
  reasoningWithoutFollowingItem?: "drop-reasoning" | "keep";
}

export type ResolvedPolicy = Required<Policy>;

export interface HealOptions {
  /**
   * Provider to heal for. Different providers have different structural rules.
   * If omitted, shared rules are applied but provider-specific rules are not.
   */
  provider?: Provider;

  /** Override any policy. Unspecified keys use safe defaults. */
  policy?: Policy;

  /**
   * If set, healMessages throws when it had to make *structural* repairs
   * (orphan tool calls, invalid tool names, etc.). Useful during development.
   */
  throwOnRepair?: boolean;

  /**
   * Optional hook invoked for every repair. Useful to log to Sentry/Datadog.
   */
  onRepair?: (repair: Repair) => void;
}

export interface HealResult {
  /** Healed messages safe to send to the provider. */
  messages: ModelMessage[];
  /** Audit log of every change made. Empty array means input was already valid. */
  repairs: Repair[];
}

export const DEFAULT_POLICY: ResolvedPolicy = {
  orphanToolUse: "stub-result",
  orphanToolResult: "drop",
  invalidToolName: "rename",
  invalidToolInput: "coerce-object",
  emptyAssistantMessage: "drop",
  duplicateToolResult: "dedupe-last",
  orphanReasoningOnlyMessage: "drop-message",
  missingReasoningSignature: "drop-reasoning",
  reasoningWithoutFollowingItem: "drop-reasoning",
};

/**
 * A rule is a pure function over the message array that returns a new array
 * plus an audit trail. Rules must be idempotent: running them twice must
 * produce the same result as running them once. This is a hard requirement
 * because healMessages() is safe to call repeatedly on persisted data.
 */
export type Rule = (
  messages: ModelMessage[],
  ctx: RuleContext,
) => { messages: ModelMessage[]; repairs: Repair[] };

export interface RuleContext {
  policy: ResolvedPolicy;
  provider: Provider | undefined;
}
