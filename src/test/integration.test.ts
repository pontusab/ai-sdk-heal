/**
 * Integration tests that push healed messages through the real AI SDK
 * surface (generateText + a mock LanguageModelV3) to verify the output is
 * structurally accepted end-to-end — not just by the standalone zod schema.
 */
import { describe, expect, test } from "bun:test";
import { MockLanguageModelV3 } from "ai/test";
import { generateText } from "ai";
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

function mockModel() {
  return new MockLanguageModelV3({
    doGenerate: async () => ({
      finishReason: "stop" as const,
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
      content: [{ type: "text" as const, text: "ok" }],
      warnings: [],
    }),
  });
}

const fixtures = {
  orphanToolUse: orphanToolUseFixture,
  orphanToolUseOutputError: orphanToolUseOutputErrorFixture,
  invalidToolInput: invalidToolInputFixture,
  invalidToolName: invalidToolNameFixture,
  duplicateToolResult: duplicateToolResultFixture,
  orphanReasoningOnly: orphanReasoningOnlyFixture,
  unsignedReasoning: unsignedReasoningFixture,
  orphanToolResult: orphanToolResultFixture,
};

describe("integration: healed messages reach the model", () => {
  for (const [name, make] of Object.entries(fixtures)) {
    test(`${name}: generateText accepts healed messages`, async () => {
      const { messages } = healMessages(make(), { provider: "anthropic" });
      const result = await generateText({
        model: mockModel(),
        messages,
      });
      expect(result.text).toBe("ok");
    });
  }
});
