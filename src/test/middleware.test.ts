/**
 * Tests that the `withHealing` / `healMiddleware` wrapper heals prompts at
 * the LanguageModelV3 layer, right before the provider is called.
 *
 * Scope: the middleware runs *after* `convertToLanguageModelPrompt`, so it
 * only catches issues the SDK tolerates but the provider rejects — invalid
 * tool names, invalid tool inputs, unsigned reasoning, duplicate tool
 * results, reasoning-without-following-item. Orphan tool-use must be healed
 * earlier via `healMessages`; the SDK's own conversion throws before we
 * ever see the prompt.
 */
import { describe, expect, test } from "bun:test";
import type { LanguageModelV3CallOptions } from "@ai-sdk/provider";
import { convertReadableStreamToArray, MockLanguageModelV3 } from "ai/test";
import { generateText, streamText } from "ai";
import { healMiddleware, withHealing } from "../middleware";
import {
  duplicateToolResultFixture,
  invalidToolInputFixture,
  invalidToolNameFixture,
  orphanToolUseFixture,
  unsignedReasoningFixture,
} from "./fixtures";
import { isToolCallPart, isToolResultPart } from "../utils";
import { healMessages } from "../heal";

type CapturedPrompt = LanguageModelV3CallOptions["prompt"];

function spyModel() {
  const captured: { generate?: CapturedPrompt; stream?: CapturedPrompt } = {};
  const model = new MockLanguageModelV3({
    provider: "anthropic.messages",
    modelId: "claude-sonnet-4-5",
    doGenerate: async ({ prompt }) => {
      captured.generate = prompt;
      return {
        finishReason: "stop",
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        content: [{ type: "text", text: "ok" }],
        warnings: [],
      };
    },
    doStream: async ({ prompt }) => {
      captured.stream = prompt;
      return {
        stream: new ReadableStream({
          start(controller) {
            controller.enqueue({ type: "text-start", id: "1" });
            controller.enqueue({ type: "text-delta", id: "1", delta: "ok" });
            controller.enqueue({ type: "text-end", id: "1" });
            controller.enqueue({
              type: "finish",
              finishReason: "stop",
              usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
            });
            controller.close();
          },
        }),
        warnings: [],
      };
    },
  });
  return { model, captured };
}

describe("withHealing — generateText", () => {
  test("renames invalid tool name in the prompt that reaches the model", async () => {
    const { model, captured } = spyModel();
    const healed = withHealing(model);

    await generateText({
      model: healed,
      messages: invalidToolNameFixture(),
    });

    const prompt = captured.generate!;
    for (const m of prompt) {
      if (m.role === "assistant") {
        for (const p of m.content) {
          if (p.type === "tool-call") {
            expect(/^[a-zA-Z0-9_-]{1,64}$/.test(p.toolName)).toBe(true);
          }
        }
      }
      if (m.role === "tool") {
        for (const p of m.content) {
          if (p.type === "tool-result") {
            expect(/^[a-zA-Z0-9_-]{1,64}$/.test(p.toolName)).toBe(true);
          }
        }
      }
    }
  });

  test("coerces string tool input to object before it reaches the model", async () => {
    const { model, captured } = spyModel();
    const healed = withHealing(model);

    await generateText({
      model: healed,
      messages: invalidToolInputFixture(),
    });

    const prompt = captured.generate!;
    for (const m of prompt) {
      if (m.role !== "assistant") continue;
      for (const p of m.content) {
        if (p.type !== "tool-call") continue;
        expect(typeof p.input).toBe("object");
        expect(p.input).not.toBeNull();
        expect(Array.isArray(p.input)).toBe(false);
      }
    }
  });

  test("de-dupes duplicate tool-results", async () => {
    const { model, captured } = spyModel();
    // Prime the history with a completed turn so convertToLanguageModelPrompt
    // doesn't flag the dangling structure. Then append a duplicate tool
    // result + a user follow-up.
    const healed = withHealing(model);
    await generateText({
      model: healed,
      messages: [
        ...duplicateToolResultFixture(),
        { role: "assistant", content: [{ type: "text", text: "done" }] },
        { role: "user", content: "thanks" },
      ],
    });

    const prompt = captured.generate!;
    const resultIds = collectToolResultIds(prompt);
    const counts = new Map<string, number>();
    for (const id of resultIds)
      counts.set(id, (counts.get(id) ?? 0) + 1);
    for (const count of counts.values()) expect(count).toBe(1);
  });

  test("auto-infers provider from wrapped model (anthropic → drops unsigned reasoning)", async () => {
    const { model, captured } = spyModel();
    const healed = withHealing(model);

    await generateText({
      model: healed,
      messages: unsignedReasoningFixture(),
    });

    const prompt = captured.generate!;
    const assistant = prompt.find((m) => m.role === "assistant");
    expect(assistant).toBeDefined();
    const hasReasoning = (assistant!.content as Array<{ type?: string }>).some(
      (p) => p.type === "reasoning",
    );
    expect(hasReasoning).toBe(false);
  });

  test("clean input passes through untouched and fires no onHealed", async () => {
    const { model, captured } = spyModel();
    const fired: string[] = [];
    const healed = withHealing(model, {
      onHealed: (e) => fired.push(...e.repairs.map((r) => r.rule)),
    });

    await generateText({
      model: healed,
      messages: [{ role: "user", content: "hi there" }],
    });
    expect(fired).toEqual([]);
    expect(captured.generate!.length).toBe(1);
  });

  test("onHealed fires exactly once per call when repairs are made", async () => {
    const { model } = spyModel();
    const events: Array<{ type: string; ruleCount: number }> = [];
    const healed = withHealing(model, {
      onHealed: (e) =>
        events.push({ type: e.type, ruleCount: e.repairs.length }),
    });

    await generateText({
      model: healed,
      messages: invalidToolNameFixture(),
    });

    expect(events.length).toBe(1);
    expect(events[0]!.type).toBe("generate");
    expect(events[0]!.ruleCount).toBeGreaterThan(0);
  });

  test("onHealed errors are swallowed and do not break the call", async () => {
    const { model } = spyModel();
    const healed = withHealing(model, {
      onHealed: () => {
        throw new Error("boom");
      },
    });

    const result = await generateText({
      model: healed,
      messages: invalidToolNameFixture(),
    });
    expect(result.text).toBe("ok");
  });
});

describe("withHealing — streamText", () => {
  test("heals prompt for streaming requests", async () => {
    const { model, captured } = spyModel();
    const healed = withHealing(model);

    const result = streamText({
      model: healed,
      messages: invalidToolNameFixture(),
    });

    const chunks = await convertReadableStreamToArray(result.textStream);
    expect(chunks.join("")).toBe("ok");

    const prompt = captured.stream!;
    for (const m of prompt) {
      if (m.role !== "assistant") continue;
      for (const p of m.content) {
        if (p.type === "tool-call") {
          expect(/^[a-zA-Z0-9_-]{1,64}$/.test(p.toolName)).toBe(true);
        }
      }
    }
  });
});

describe("healMiddleware — manual composition", () => {
  test("can be composed via wrapLanguageModel", async () => {
    const { wrapLanguageModel } = await import("ai");
    const { model, captured } = spyModel();

    const wrapped = wrapLanguageModel({
      model,
      middleware: healMiddleware({ provider: "anthropic" }),
    });

    await generateText({
      model: wrapped,
      messages: invalidToolNameFixture(),
    });

    for (const m of captured.generate!) {
      if (m.role !== "assistant") continue;
      for (const p of m.content) {
        if (p.type === "tool-call") {
          expect(/^[a-zA-Z0-9_-]{1,64}$/.test(p.toolName)).toBe(true);
        }
      }
    }
  });
});

describe("withHealing — documented limits", () => {
  /**
   * The AI SDK validates tool-call/result pairing during
   * `convertToLanguageModelPrompt`, which runs before any middleware. That
   * means orphan tool-use is rejected by the SDK itself (throws
   * `MissingToolResultsError`) before the middleware gets a chance.
   *
   * Callers who need to repair orphan tool-use must run `healMessages` on
   * the message array before passing it to `generateText` / `streamText`.
   * `withHealing` alone is not enough.
   */
  test("orphan tool-use is still rejected because SDK conversion runs first", async () => {
    const { model } = spyModel();
    const healed = withHealing(model);

    await expect(
      generateText({
        model: healed,
        messages: orphanToolUseFixture(),
      }),
    ).rejects.toThrow(/AI_MissingToolResultsError|Tool result is missing/);
  });

  test("…but `healMessages` → `generateText` composed with `withHealing` handles everything", async () => {
    const { model } = spyModel();
    const healed = withHealing(model);
    const { messages } = healMessages(orphanToolUseFixture(), {
      provider: "anthropic",
    });
    const result = await generateText({ model: healed, messages });
    expect(result.text).toBe("ok");
  });
});

// ─── helpers ────────────────────────────────────────────────────────────

function collectToolResultIds(prompt: CapturedPrompt): string[] {
  const out: string[] = [];
  for (const m of prompt) {
    if (m.role !== "tool") continue;
    for (const p of m.content as Array<unknown>) {
      if (isToolResultPart(p)) out.push(p.toolCallId);
    }
  }
  return out;
}

// Keep unused imports referenced so the linter sees the intent (used in
// scaffolding, may be re-enabled in future tests).
void isToolCallPart;
