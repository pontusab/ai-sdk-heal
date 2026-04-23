# Changelog

## 0.2.0

- `withHealing(model, options?)` — drop-in wrapper that heals every prompt
  the model sees through an AI SDK middleware. Provider is auto-detected
  from the underlying model; override via `options.provider`.
- `healMiddleware(options?)` — the same logic exposed as a
  `LanguageModelV3Middleware` you can compose manually through
  `wrapLanguageModel`.
- `validateMessages(messages, options?)` — non-mutating check that returns
  `{ valid, issues }` using the same detection pipeline. Useful in tests
  or CI assertions.
- Documented scope: middleware heals provider-rejection issues (invalid
  tool names / inputs, unsigned reasoning, duplicate tool results,
  trailing reasoning). Orphan tool-use must still be fixed with
  `healMessages` because the SDK validates pairing during its own
  prompt conversion.

## 0.1.0

First release.

- `healMessages(messages, options)` — pure, idempotent message array healer
- `inferProvider(model)` — auto-detect provider from an AI SDK model instance
- `MessageHealingError` — thrown when `throwOnRepair: true`
- Individual rules exported for custom pipelines
- Providers: shared rules + Anthropic-specific + OpenAI-specific
- Full test coverage against the exact message shapes from tracked upstream
  scenarios (tool-use/result pairing, invalid tool names, invalid tool inputs,
  reasoning-only messages, unsigned reasoning, trailing reasoning, duplicate
  and orphan tool results)
- Integration tests using `MockLanguageModelV3` to verify healed output flows
  through the real `generateText` path
