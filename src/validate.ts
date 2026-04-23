import type { ModelMessage } from "ai";
import { healMessages } from "./heal";
import type { HealOptions, Repair } from "./types";

export interface ValidateResult {
  /** `true` when no repairs would be required to send this array to the provider. */
  valid: boolean;
  /**
   * Every issue the healer would fix. Empty when `valid` is `true`. The
   * entries are the same {@link Repair} objects `healMessages` emits, so you
   * can route them straight to your telemetry pipeline.
   */
  issues: Repair[];
}

/**
 * Check whether a message array is ready to send to the provider without
 * mutating anything.
 *
 * `validateMessages` runs the same detection logic as {@link healMessages}
 * but discards the healed output, so it's cheap to call in development or
 * in test assertions.
 *
 * @example
 *
 *   const { valid, issues } = validateMessages(messages, { provider: "anthropic" });
 *   if (!valid) {
 *     console.warn("messages will be healed before send:", issues);
 *   }
 *
 * @example Guard in a test:
 *
 *   expect(validateMessages(messages, { provider: "openai" }).valid).toBe(true);
 */
export function validateMessages(
  input: ModelMessage[],
  options: HealOptions = {},
): ValidateResult {
  // Force-throw is a healing concern, not a validation concern — strip it
  // so `validateMessages` always returns cleanly.
  const { throwOnRepair: _throwOnRepair, ...rest } = options;
  const { repairs } = healMessages(input, rest);
  return {
    valid: repairs.length === 0,
    issues: repairs,
  };
}
