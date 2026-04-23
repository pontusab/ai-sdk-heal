import type {
  LanguageModelV3,
  LanguageModelV3Middleware,
  LanguageModelV3Prompt,
} from "@ai-sdk/provider";
import type { ModelMessage } from "ai";
import { wrapLanguageModel } from "ai";
import { healMessages, inferProvider } from "./heal";
import type { HealOptions, HealResult, Provider, Repair } from "./types";

/**
 * Options for the healing middleware. Same as {@link HealOptions} plus a
 * hook that lets callers observe when healing modified the prompt.
 */
export interface HealMiddlewareOptions extends HealOptions {
  /**
   * Called with every batch of repairs applied to a prompt before it reaches
   * the provider. Fires once per generate/stream call when at least one
   * repair was made. Use it to log model-visible corrections to your
   * telemetry pipeline (Sentry / Datadog / logs).
   */
  onHealed?: (event: {
    type: "generate" | "stream";
    repairs: Repair[];
    model: LanguageModelV3;
  }) => void;
}

/**
 * Create a {@link LanguageModelV3Middleware} that heals the prompt just
 * before it reaches the provider. Detection and repair use the same rules
 * as {@link healMessages}.
 *
 * Prefer {@link withHealing} unless you need to compose it manually with
 * other middleware via `wrapLanguageModel`.
 *
 * @example
 *
 *   const model = wrapLanguageModel({
 *     model: anthropic("claude-sonnet-4-5"),
 *     middleware: healMiddleware({ throwOnRepair: false }),
 *   });
 */
export function healMiddleware(
  options: HealMiddlewareOptions = {},
): LanguageModelV3Middleware {
  const { onHealed, ...healOptions } = options;

  return {
    specificationVersion: "v3",
    transformParams: async ({ params, type, model }) => {
      const provider = healOptions.provider ?? inferProvider(model);

      // LanguageModelV3Prompt is a structural subset of ModelMessage[] for
      // the fields the rules care about (text, reasoning, tool-call,
      // tool-result, providerOptions). The cast is safe for the scope of
      // healing and we cast back on output.
      const asMessages = params.prompt as unknown as ModelMessage[];
      const { messages, repairs }: HealResult = healMessages(asMessages, {
        ...healOptions,
        provider,
      });

      if (repairs.length === 0) return params;

      if (onHealed) {
        try {
          onHealed({ type, repairs, model });
        } catch {
          // Never let an observability hook break the request.
        }
      }

      return {
        ...params,
        prompt: messages as unknown as LanguageModelV3Prompt,
      };
    },
  };
}

/**
 * Wrap a language model so every prompt sent to it is healed first.
 *
 * This is the drop-in recommended path for production apps: you wrap the
 * model once at configuration time and every `generateText` / `streamText`
 * call automatically gets orphan tool calls repaired, invalid tool names
 * sanitised, and provider-specific reasoning constraints enforced.
 *
 * The provider is auto-detected from the underlying model. Override via
 * `options.provider` for exotic setups (custom gateway, Bedrock rewrapping,
 * etc.).
 *
 * @example
 *
 *   import { anthropic } from "@ai-sdk/anthropic";
 *   import { withHealing } from "ai-sdk-heal";
 *
 *   const model = withHealing(anthropic("claude-sonnet-4-5"));
 *
 *   // every call to generateText / streamText is now self-healing
 *   await streamText({ model, messages });
 */
export function withHealing(
  model: LanguageModelV3,
  options: HealMiddlewareOptions = {},
): LanguageModelV3 {
  const provider: Provider | undefined =
    options.provider ?? inferProvider(model);

  return wrapLanguageModel({
    model,
    middleware: healMiddleware({ ...options, provider }),
  });
}
