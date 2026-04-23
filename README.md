# ai-sdk-heal

Keep your AI SDK conversations valid. `ai-sdk-heal` normalizes message arrays so they satisfy each provider's structural rules — pairing tool calls with results, coercing tool inputs to objects, preserving reasoning blocks correctly, and more.

One function. Pure. Idempotent. Safe on the hot path and on persisted history.

```ts
import { healMessages } from "ai-sdk-heal";

const { messages, repairs } = healMessages(rawMessages, { provider: "anthropic" });
await streamText({ model, messages });
```

## What it does

Providers each have their own rules for what a valid message history looks like:

- Anthropic requires every `tool_use` to be paired with a matching `tool_result`, reasoning blocks to carry a `signature`, and rejects assistant messages that contain only reasoning.
- OpenAI's Responses API expects reasoning items to be followed by a same-flow item.
- All of them require tool inputs to be objects and tool names to match `^[a-zA-Z0-9_-]{1,64}$`.

Agents, retries, and persisted conversations make it easy to drift out of those rules — especially across long multi-turn flows with thinking models and parallel tool calls. `ai-sdk-heal` checks the whole history against the active provider's rules and returns a normalized copy, plus an audit trail of every change it made.

## Rules

| Rule | What it does |
|---|---|
| `orphan-tool-use` | Assistant `tool-call` with no matching `tool-result`: inserts a placeholder result (or drops the call) so the pairing invariant holds |
| `orphan-tool-result` | `tool-result` referencing a call that isn't in history: drops it |
| `invalid-tool-input` | Tool input stored as a raw string because JSON parsing failed upstream: coerces to `{ raw: "…" }` so subsequent turns stay usable |
| `invalid-tool-name` | Tool names with characters outside `^[a-zA-Z0-9_-]{1,64}$`: sanitizes while keeping the call/result pair linked |
| `duplicate-tool-result` | Same `toolCallId` appearing twice after a retry: dedupes |
| `empty-assistant-message` | Assistant message with no substantive content: drops it |
| `orphan-reasoning-only-message` (Anthropic) | After pruning, an assistant message contains only reasoning blocks: drops it |
| `missing-reasoning-signature` (Anthropic) | Reasoning block with no `providerOptions.anthropic.signature`: drops it (Anthropic won't accept thinking without the signature on replay) |
| `reasoning-without-following-item` (OpenAI) | Trailing reasoning part with no following item in the Responses flow: drops it |

Every change is captured in the `repairs` array so you can log it, alert on it, or surface it in admin tooling.

Each rule maps to a documented scenario tracked upstream: [#8516](https://github.com/vercel/ai/issues/8516), [#9141](https://github.com/vercel/ai/issues/9141), [#11602](https://github.com/vercel/ai/issues/11602), [#13430](https://github.com/vercel/ai/issues/13430), [#13645](https://github.com/vercel/ai/issues/13645), [#14259](https://github.com/vercel/ai/issues/14259), [#8379](https://github.com/vercel/ai/issues/8379), [#7729](https://github.com/vercel/ai/issues/7729), [#12504](https://github.com/vercel/ai/issues/12504).

## Install

```bash
npm install ai-sdk-heal
```

Peer dependency: `ai >= 5.0`.

## Usage

### Heal before the provider call

```ts
import { healMessages } from "ai-sdk-heal";
import { anthropic } from "@ai-sdk/anthropic";
import { streamText } from "ai";

const { messages, repairs } = healMessages(rawMessages, {
  provider: "anthropic",
  onRepair: (r) => logger.info({ repair: r }, "message-normalized"),
});

const result = streamText({
  model: anthropic("claude-sonnet-4-20250514"),
  messages,
});
```

If you want to hard-fail during development instead:

```ts
healMessages(rawMessages, { provider: "anthropic", throwOnRepair: true });
```

### Wrap your model once with `withHealing`

If you'd rather not remember to call `healMessages` on every request, wrap
the model itself. The wrapper heals the prompt as it passes through the AI
SDK middleware layer:

```ts
import { withHealing } from "ai-sdk-heal";
import { anthropic } from "@ai-sdk/anthropic";

const model = withHealing(anthropic("claude-sonnet-4-5"), {
  onHealed: ({ repairs }) =>
    logger.warn({ repairs }, "prompt-auto-healed"),
});

// Every generateText / streamText call now gets auto-healed.
await streamText({ model, messages });
```

Provider is auto-detected from the underlying model; override via
`{ provider: "anthropic" }` for custom gateways.

**Scope.** The middleware runs after the AI SDK's prompt conversion, so it
handles issues only the provider would reject — invalid tool names,
malformed tool inputs, unsigned reasoning, duplicate tool results,
reasoning-without-following-item. **Orphan tool calls** still need
`healMessages` up-front, because the SDK validates pairing during its own
`convertToLanguageModelPrompt` pass. A robust setup combines both:

```ts
const healedMessages = healMessages(rawMessages, { provider: "anthropic" }).messages;
await streamText({ model: withHealing(anthropic("claude-sonnet-4-5")), messages: healedMessages });
```

You can also compose `healMiddleware` manually via `wrapLanguageModel`:

```ts
import { wrapLanguageModel } from "ai";
import { healMiddleware } from "ai-sdk-heal";

const model = wrapLanguageModel({
  model: anthropic("claude-sonnet-4-5"),
  middleware: [healMiddleware(), otherMiddleware()],
});
```

### Validate without mutating

Use `validateMessages` in tests or CI to assert a conversation is
provider-ready without changing it:

```ts
import { validateMessages } from "ai-sdk-heal";

const { valid, issues } = validateMessages(messages, { provider: "anthropic" });
if (!valid) {
  // `issues` is the same `Repair[]` shape healMessages returns.
  throw new Error(`conversation is not provider-ready: ${issues.map((i) => i.rule).join(", ")}`);
}
```

### Heal persisted conversations

Because `healMessages` is idempotent — running it twice produces the same result — it's safe to apply on every read, or as a one-shot migration:

```ts
import { healMessages } from "ai-sdk-heal";

for await (const row of db.selectFrom("chat").execute()) {
  const { messages, repairs } = healMessages(row.messages, {
    provider: row.provider,
  });
  if (repairs.length === 0) continue;
  await db
    .updateTable("chat")
    .set({ messages, healed_at: new Date() })
    .where("id", "=", row.id)
    .execute();
}
```

### Auto-detect the provider

```ts
import { healMessages, inferProvider } from "ai-sdk-heal";

const provider = inferProvider(model);
const { messages } = healMessages(rawMessages, { provider });
```

## Policies

Every rule has a default action picked to keep conversations usable. Override any of them:

```ts
healMessages(rawMessages, {
  provider: "anthropic",
  policy: {
    orphanToolUse: "drop-call",          // default: "stub-result"
    invalidToolName: "drop-pair",         // default: "rename"
    invalidToolInput: "empty-object",     // default: "coerce-object"
    duplicateToolResult: "dedupe-first",  // default: "dedupe-last"
    missingReasoningSignature: "keep",    // default: "drop-reasoning"
  },
});
```

See `Policy` in the types for every option.

## Design

- **Pure and idempotent.** No side effects, no I/O. Running `heal(heal(x))` always equals `heal(x)` — this is enforced in the test suite and makes the package safe to apply unconditionally.
- **Provider-aware.** Shared rules run for every provider; provider-specific rules (Anthropic, OpenAI) layer on top.
- **Auditable.** Every change returns a `Repair` record with the rule name, message index, and reason.
- **Composable.** Individual rules are exported so you can build your own pipeline.

## Related

- [`toolpick`](https://github.com/pontusab/toolpick) — dynamic tool selection for the AI SDK so the model only sees the tools that matter on each step.

## License

MIT
