import type { Repair, Rule } from "../types";
import {
  isAssistant,
  isReasoningPart,
  replaceAssistantContent,
} from "../utils";

/**
 * OpenAI's Responses API rejects a `reasoning` item that isn't followed by a
 * same-flow item (vercel/ai#8379). In ModelMessage terms, this maps to a
 * reasoning part that is the last content in an assistant message with no
 * following text or tool-call part, and no text/tool-call in the next
 * assistant message of the same turn.
 *
 * We apply the same conservative fix as Anthropic: drop trailing reasoning
 * parts that have no following content in the same message.
 */
export const healReasoningWithoutFollowingItem: Rule = (
  messages,
  { policy },
) => {
  if (policy.reasoningWithoutFollowingItem === "keep")
    return { messages, repairs: [] };

  const repairs: Repair[] = [];
  const next = messages.map((m, mi) => {
    if (!isAssistant(m) || typeof m.content === "string") return m;
    const content = [...m.content];
    let changed = false;
    // Walk from the end, dropping reasoning parts until we hit something else.
    while (content.length > 0) {
      const last = content[content.length - 1];
      if (last && isReasoningPart(last)) {
        repairs.push({
          rule: "reasoning-without-following-item",
          messageIndex: mi,
          partIndex: content.length - 1,
          action: "dropped-part",
          reason:
            "Trailing reasoning part without a following text/tool-call item; OpenAI responses API rejects this",
        });
        content.pop();
        changed = true;
        continue;
      }
      break;
    }
    return changed ? replaceAssistantContent(m, content) : m;
  });
  return { messages: next, repairs };
};
