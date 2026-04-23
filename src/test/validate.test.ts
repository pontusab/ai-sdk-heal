import { describe, expect, test } from "bun:test";
import { validateMessages } from "../validate";
import {
  duplicateToolResultFixture,
  invalidToolInputFixture,
  invalidToolNameFixture,
  orphanReasoningOnlyFixture,
  orphanToolResultFixture,
  orphanToolUseFixture,
  unsignedReasoningFixture,
} from "./fixtures";

describe("validateMessages", () => {
  test("flags orphan tool calls", () => {
    const result = validateMessages(orphanToolUseFixture(), {
      provider: "anthropic",
    });
    expect(result.valid).toBe(false);
    expect(result.issues.some((r) => r.rule === "orphan-tool-use")).toBe(true);
  });

  test("flags invalid tool names", () => {
    const result = validateMessages(invalidToolNameFixture(), {
      provider: "anthropic",
    });
    expect(result.valid).toBe(false);
    expect(result.issues.some((r) => r.rule === "invalid-tool-name")).toBe(true);
  });

  test("flags invalid tool inputs", () => {
    const result = validateMessages(invalidToolInputFixture(), {
      provider: "anthropic",
    });
    expect(result.valid).toBe(false);
    expect(result.issues.some((r) => r.rule === "invalid-tool-input")).toBe(
      true,
    );
  });

  test("flags duplicate tool results", () => {
    const result = validateMessages(duplicateToolResultFixture());
    expect(result.valid).toBe(false);
    expect(result.issues.some((r) => r.rule === "duplicate-tool-result")).toBe(
      true,
    );
  });

  test("flags orphan tool results", () => {
    const result = validateMessages(orphanToolResultFixture());
    expect(result.valid).toBe(false);
    expect(result.issues.some((r) => r.rule === "orphan-tool-result")).toBe(
      true,
    );
  });

  test("flags reasoning-only assistant (anthropic only)", () => {
    const anthropic = validateMessages(orphanReasoningOnlyFixture(), {
      provider: "anthropic",
    });
    expect(anthropic.valid).toBe(false);
    expect(
      anthropic.issues.some((r) => r.rule === "orphan-reasoning-only-message"),
    ).toBe(true);

    // No provider set — shared rules only, so reasoning-only is not flagged.
    const nonSpecific = validateMessages(orphanReasoningOnlyFixture());
    expect(nonSpecific.valid).toBe(true);
  });

  test("flags unsigned reasoning (anthropic only)", () => {
    const result = validateMessages(unsignedReasoningFixture(), {
      provider: "anthropic",
    });
    expect(result.valid).toBe(false);
    expect(
      result.issues.some((r) => r.rule === "missing-reasoning-signature"),
    ).toBe(true);
  });

  test("returns valid for clean input", () => {
    const result = validateMessages([
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "t1",
            toolName: "search",
            input: { q: "ok" },
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "t1",
            toolName: "search",
            output: { type: "text", value: "done" },
          },
        ],
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "here you go" }],
      },
    ], { provider: "anthropic" });
    expect(result.valid).toBe(true);
    expect(result.issues).toEqual([]);
  });

  test("does not mutate the input array", () => {
    const input = orphanToolUseFixture();
    const snapshot = JSON.stringify(input);
    validateMessages(input, { provider: "anthropic" });
    expect(JSON.stringify(input)).toBe(snapshot);
  });

  test("never throws even if throwOnRepair is set in options", () => {
    // throwOnRepair is a healing concern; validate should ignore it.
    expect(() =>
      validateMessages(orphanToolUseFixture(), {
        provider: "anthropic",
        throwOnRepair: true,
      }),
    ).not.toThrow();
  });

  test("respects policy overrides (keep means no issue reported)", () => {
    const result = validateMessages(orphanToolUseFixture(), {
      provider: "anthropic",
      policy: { orphanToolUse: "keep" },
    });
    // With `keep`, we still emit a single orphan-tool-use record for
    // visibility (so users know the provider will reject). Only in
    // `stub-result`/`drop-call` are we actually fixing.
    expect(
      result.issues.some((r) => r.rule === "orphan-tool-use"),
    ).toBe(true);
  });
});
