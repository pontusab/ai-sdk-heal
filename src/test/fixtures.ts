import type { ModelMessage } from "ai";

/**
 * Each fixture is lifted from a real issue on vercel/ai. The comment at the
 * top of each case links to the bug so regressions stay traceable.
 */

// vercel/ai#8516: assistant tool_use with no matching tool_result, then a new user turn.
export const orphanToolUseFixture = (): ModelMessage[] => [
  { role: "user", content: "generate 10 items" },
  {
    role: "assistant",
    content: [
      {
        type: "tool-call",
        toolCallId: "tool-example-123",
        toolName: "json",
        input: { message: "generate 10 items" },
      },
      { type: "text", text: "I generated code for 10 items." },
    ],
  },
  { role: "user", content: "generate 100 items" },
];

// vercel/ai#14259: adapter produces orphaned tool_use for output-error state tools.
export const orphanToolUseOutputErrorFixture = (): ModelMessage[] => [
  { role: "user", content: "search for puppies" },
  {
    role: "assistant",
    content: [
      {
        type: "tool-call",
        toolCallId: "call_abc",
        toolName: "web_search",
        input: { q: "puppies" },
      },
    ],
  },
  { role: "user", content: "actually never mind" },
];

// vercel/ai#13645: tool_use.input is a string (rawInput) instead of an object.
export const invalidToolInputFixture = (): ModelMessage[] => [
  { role: "user", content: "send an email" },
  {
    role: "assistant",
    content: [
      {
        type: "tool-call",
        toolCallId: "call_xyz",
        toolName: "send_email",
        // ⚠ This is the shape that bricks Anthropic 400s. Real AI SDK
        // output when model emits invalid JSON for tool input.
        input: '{"to": John Doe',
      },
    ],
  },
  {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: "call_xyz",
        toolName: "send_email",
        output: { type: "error-text", value: "invalid input" },
      },
    ],
  },
];

// vercel/ai#9141: model hallucinated XML as tool name.
export const invalidToolNameFixture = (): ModelMessage[] => [
  { role: "user", content: "project data" },
  {
    role: "assistant",
    content: [
      {
        type: "tool-call",
        toolCallId: "tooluse_Wi",
        toolName: 'getProjectDataName" />\n \n \n \n ',
        input: {},
      },
    ],
  },
  {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: "tooluse_Wi",
        toolName: 'getProjectDataName" />\n \n \n \n ',
        output: { type: "error-text", value: "unavailable" },
      },
    ],
  },
];

// vercel/ai#13430: after pruneMessages, an assistant message has only a
// reasoning block. Anthropic rejects reasoning-only messages.
export const orphanReasoningOnlyFixture = (): ModelMessage[] => [
  { role: "user", content: "what's broken?" },
  {
    role: "assistant",
    content: [
      {
        type: "reasoning",
        text: "Let me check for errors...",
        providerOptions: { anthropic: { signature: "abc" } },
      },
    ],
  },
  { role: "user", content: "anything?" },
];

// vercel/ai#11602 / #7729: reasoning without signature must be dropped.
export const unsignedReasoningFixture = (): ModelMessage[] => [
  { role: "user", content: "think about this" },
  {
    role: "assistant",
    content: [
      { type: "reasoning", text: "hmm the user wants..." },
      { type: "text", text: "here you go" },
    ],
  },
];

// Duplicate tool result caused by bad retry.
export const duplicateToolResultFixture = (): ModelMessage[] => [
  {
    role: "assistant",
    content: [
      {
        type: "tool-call",
        toolCallId: "dup",
        toolName: "search",
        input: { q: "x" },
      },
    ],
  },
  {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: "dup",
        toolName: "search",
        output: { type: "text", value: "first" },
      },
    ],
  },
  {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: "dup",
        toolName: "search",
        output: { type: "text", value: "second (retried)" },
      },
    ],
  },
];

// Orphan tool_result: a tool message references an id we've never seen.
export const orphanToolResultFixture = (): ModelMessage[] => [
  { role: "user", content: "hi" },
  {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: "nonexistent",
        toolName: "search",
        output: { type: "text", value: "result" },
      },
    ],
  },
];
