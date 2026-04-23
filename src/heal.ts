import type { ModelMessage } from "ai";
import { rulesFor } from "./rules";
import {
  DEFAULT_POLICY,
  type HealOptions,
  type HealResult,
  type Provider,
  type Repair,
} from "./types";

/**
 * Heal a message array so it's safe to send to the provider.
 *
 * Rules are idempotent: calling `healMessages` twice on the same input
 * produces the same result. That's important — you can safely call it both
 * on the hot path before the provider request and offline against persisted
 * DB rows that may have been bricked by older SDK versions.
 *
 * @example
 *
 *   const { messages, repairs } = healMessages(raw, { provider: "anthropic" });
 *   if (repairs.length) logger.warn({ repairs }, "message-history-healed");
 *   await streamText({ model, messages });
 */
export function healMessages(
  input: ModelMessage[],
  options: HealOptions = {},
): HealResult {
  const policy = { ...DEFAULT_POLICY, ...options.policy };
  const ctx = { policy, provider: options.provider };
  const rules = rulesFor(options.provider);

  let messages = input;
  const allRepairs: Repair[] = [];

  for (const rule of rules) {
    const { messages: next, repairs } = rule(messages, ctx);
    messages = next;
    if (repairs.length === 0) continue;
    for (const r of repairs) {
      allRepairs.push(r);
      if (options.onRepair) {
        try {
          options.onRepair(r);
        } catch {
          // Never let an observability hook break healing.
        }
      }
    }
  }

  if (options.throwOnRepair && allRepairs.length > 0) {
    throw new MessageHealingError(allRepairs);
  }

  return { messages, repairs: allRepairs };
}

/**
 * Infer the provider id from a language model instance. Returns `undefined`
 * if the provider isn't recognised — the caller can fall back to shared
 * rules only.
 */
export function inferProvider(
  model: unknown,
): Provider | undefined {
  if (!model || typeof model !== "object") return undefined;
  const m = model as { provider?: string; modelId?: string };
  const id = typeof m.provider === "string" ? m.provider.toLowerCase() : "";
  if (id.includes("anthropic") && id.includes("bedrock"))
    return "bedrock-anthropic";
  if (id.includes("anthropic")) return "anthropic";
  if (id.includes("openai")) return "openai";
  if (id.includes("google") || id.includes("vertex") || id.includes("gemini"))
    return "google";
  return undefined;
}

export class MessageHealingError extends Error {
  readonly repairs: Repair[];
  constructor(repairs: Repair[]) {
    super(
      `ai-sdk-heal repaired ${repairs.length} issue${
        repairs.length === 1 ? "" : "s"
      }: ${repairs
        .slice(0, 3)
        .map((r) => r.rule)
        .join(", ")}${repairs.length > 3 ? "…" : ""}`,
    );
    this.name = "MessageHealingError";
    this.repairs = repairs;
  }
}
