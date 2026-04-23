import type {
  AssistantContent,
  AssistantModelMessage,
  ModelMessage,
  TextPart,
  ToolCallPart,
  ToolContent,
  ToolModelMessage,
  ToolResultPart,
  UserContent,
} from "ai";

/**
 * Subset of the AI SDK's `ReasoningPart` we rely on. Defined locally so we
 * don't force consumers to add `@ai-sdk/provider-utils` just for the type.
 */
export interface ReasoningPart {
  type: "reasoning";
  text: string;
  providerOptions?: Record<string, unknown>;
}

export const isAssistant = (m: ModelMessage): m is AssistantModelMessage =>
  m.role === "assistant";

export const isTool = (m: ModelMessage): m is ToolModelMessage =>
  m.role === "tool";

export const isUser = (m: ModelMessage): boolean => m.role === "user";

/** Normalized view: string content coerced to a single text part. */
export function assistantParts(
  m: AssistantModelMessage,
): Exclude<AssistantContent, string> {
  if (typeof m.content === "string") {
    return m.content.length === 0
      ? []
      : ([{ type: "text", text: m.content }] as Exclude<AssistantContent, string>);
  }
  return m.content;
}

export function userParts(
  content: UserContent,
): Exclude<UserContent, string> {
  if (typeof content === "string") {
    return content.length === 0
      ? []
      : ([{ type: "text", text: content }] as Exclude<UserContent, string>);
  }
  return content;
}

export function toolParts(m: ToolModelMessage): ToolContent {
  return m.content;
}

export function isTextPart(p: unknown): p is TextPart {
  return !!p && typeof p === "object" && (p as { type?: string }).type === "text";
}

export function isReasoningPart(p: unknown): p is ReasoningPart {
  return (
    !!p && typeof p === "object" && (p as { type?: string }).type === "reasoning"
  );
}

export function isToolCallPart(p: unknown): p is ToolCallPart {
  return (
    !!p && typeof p === "object" && (p as { type?: string }).type === "tool-call"
  );
}

export function isToolResultPart(p: unknown): p is ToolResultPart {
  return (
    !!p &&
    typeof p === "object" &&
    (p as { type?: string }).type === "tool-result"
  );
}

/** Tool name validity per Anthropic / Bedrock: ^[a-zA-Z0-9_-]{1,64}$ */
export const TOOL_NAME_RE = /^[a-zA-Z0-9_-]{1,64}$/;

export function sanitizeToolName(name: string): string {
  const cleaned = name.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
  return cleaned.length === 0 ? "unknown_tool" : cleaned;
}

/** Has any non-whitespace, non-reasoning content? */
export function hasSubstantiveContent(content: AssistantContent): boolean {
  if (typeof content === "string") return content.trim().length > 0;
  for (const p of content) {
    if (isTextPart(p) && p.text.trim().length > 0) return true;
    if (isToolCallPart(p)) return true;
    if (isToolResultPart(p)) return true;
    if (!!p && typeof p === "object" && (p as { type?: string }).type === "file")
      return true;
    if (
      !!p &&
      typeof p === "object" &&
      (p as { type?: string }).type === "tool-approval-request"
    )
      return true;
  }
  return false;
}

/** Is an assistant message empty or reasoning-only? */
export function isReasoningOnlyAssistant(m: AssistantModelMessage): boolean {
  if (typeof m.content === "string") return false;
  if (m.content.length === 0) return false;
  const hasReasoning = m.content.some(isReasoningPart);
  if (!hasReasoning) return false;
  const hasOther = m.content.some((p) => !isReasoningPart(p));
  return !hasOther;
}

/** Shallow-clone an assistant/tool message with new content. */
export function replaceAssistantContent(
  m: AssistantModelMessage,
  content: AssistantContent,
): AssistantModelMessage {
  return { ...m, content };
}

export function replaceToolContent(
  m: ToolModelMessage,
  content: ToolContent,
): ToolModelMessage {
  return { ...m, content };
}
