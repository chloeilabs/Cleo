import "server-only";

import type { ConversationTurn, SDKMessage } from "@cursor/sdk";

import { buildToolView } from "@/lib/server/tools";
import type { TimelineItem } from "@/lib/types";

/**
 * Two sources produce the same timeline: `run.conversation()` for a settled
 * transcript and `run.stream()` for a live one. Both funnel through here so the
 * UI never has to know which one it is looking at.
 */

/** Convert a settled transcript into timeline items. */
export function itemsFromConversation(
  turns: ConversationTurn[],
): TimelineItem[] {
  const items: TimelineItem[] = [];

  turns.forEach((entry, turnIndex) => {
    if (entry.type === "shellConversationTurn") {
      const { shellCommand, shellOutput } = entry.turn;
      if (!shellCommand && !shellOutput) return;

      items.push({
        id: `c${turnIndex}:shell`,
        kind: "shell",
        command: shellCommand?.command ?? "",
        workingDirectory: shellCommand?.workingDirectory,
        stdout: shellOutput?.stdout,
        stderr: shellOutput?.stderr,
        exitCode: shellOutput?.exitCode,
      });
      return;
    }

    const { userMessage, steps } = entry.turn;

    if (userMessage?.text) {
      items.push({
        id: `c${turnIndex}:user`,
        kind: "user",
        text: userMessage.text,
      });
    }

    steps.forEach((step, stepIndex) => {
      const id = `c${turnIndex}:${stepIndex}`;

      switch (step.type) {
        case "assistantMessage": {
          if (!step.message.text) break;
          items.push({ id, kind: "assistant", text: step.message.text });
          break;
        }
        case "thinkingMessage": {
          if (!step.message.text) break;
          items.push({
            id,
            kind: "thinking",
            text: step.message.text,
            durationMs: step.message.thinkingDurationMs,
          });
          break;
        }
        case "toolCall": {
          const call = step.message;
          items.push({
            id,
            kind: "tool",
            callId: id,
            tool: buildToolView(
              call.type,
              call.result === undefined ? "running" : "completed",
              call.args,
              call.result,
            ),
          });
          break;
        }
      }
    });
  });

  return items;
}

/**
 * Convert one live stream event into a timeline item.
 *
 * Items carry stable ids so the client can upsert: a tool call reuses its
 * `call_id` across the `running` and `completed` events, and repeated cloud
 * lifecycle transitions collapse onto a single row.
 */
export function itemFromStreamEvent(
  event: SDKMessage,
  sequence: number,
): TimelineItem | undefined {
  switch (event.type) {
    case "user": {
      const text = event.message.content
        .map((block) => block.text)
        .join("")
        .trim();
      return text ? { id: `s${sequence}:user`, kind: "user", text } : undefined;
    }

    case "assistant": {
      const text = event.message.content
        .flatMap((block) => (block.type === "text" ? [block.text] : []))
        .join("")
        .trim();
      return text
        ? { id: `s${sequence}:assistant`, kind: "assistant", text }
        : undefined;
    }

    case "thinking": {
      return event.text
        ? {
            id: `s${sequence}:thinking`,
            kind: "thinking",
            text: event.text,
            durationMs: event.thinking_duration_ms,
          }
        : undefined;
    }

    case "tool_call": {
      return {
        id: `tool:${event.call_id}`,
        kind: "tool",
        callId: event.call_id,
        tool: buildToolView(
          event.name,
          event.status,
          event.args,
          event.result,
          Boolean(event.truncated?.args || event.truncated?.result),
        ),
      };
    }

    case "status": {
      return {
        id: `status:${event.status}`,
        kind: "status",
        status: event.status,
        message: event.message,
      };
    }

    case "usage": {
      return { id: `s${sequence}:usage`, kind: "usage", usage: event.usage };
    }

    // `system` carries init metadata and `task`/`request` are milestone pings;
    // neither adds anything the timeline renders.
    default:
      return undefined;
  }
}

/** Kinds the live stream produces that a stored transcript has no notion of. */
const STREAM_ONLY = new Set<TimelineItem["kind"]>(["status", "usage", "error"]);

/**
 * A rough identity for a row, used to tell whether a streamed item and a stored
 * one describe the same moment. Content is compared loosely because the stored
 * copy is canonical and may differ in whitespace or truncation.
 */
function signature(item: TimelineItem): string {
  switch (item.kind) {
    case "tool":
      return `tool:${item.tool.name}:${item.tool.title}`;
    case "user":
    case "assistant":
    case "thinking":
      return `${item.kind}:${item.text.slice(0, 48)}`;
    case "shell":
      return `shell:${item.command}`;
    default:
      return item.kind;
  }
}

/**
 * Fold the run's stored transcript into what was already streamed.
 *
 * The stored copy wins on content — it is what Cursor actually persisted — but
 * rows that clearly correspond keep the id they were streamed under, so the
 * client's timeline settles without remounting components and losing, say, an
 * expanded diff. Cloud lifecycle rows exist only on the stream, so they are
 * carried across explicitly.
 */
export function mergeTranscript(
  rawStreamed: TimelineItem[],
  stored: TimelineItem[],
): TimelineItem[] {
  // A tool call is emitted twice — once running, once complete — so collapse
  // repeats onto the position where the id first appeared.
  const seen = new Map<string, number>();
  const streamed: TimelineItem[] = [];

  for (const item of rawStreamed) {
    const index = seen.get(item.id);
    if (index === undefined) {
      seen.set(item.id, streamed.length);
      streamed.push(item);
    } else {
      streamed[index] = item;
    }
  }

  if (streamed.length === 0) return stored;

  // Remember each lifecycle row by how many content rows preceded it, so it can
  // be spliced back into the same place in the reconciled list.
  const content: TimelineItem[] = [];
  const lifecycle: Array<{ after: number; item: TimelineItem }> = [];

  for (const item of streamed) {
    if (item.kind === "status") {
      lifecycle.push({ after: content.length, item });
    } else if (!STREAM_ONLY.has(item.kind)) {
      content.push(item);
    }
  }

  const reconciled = stored.map((item, index) => {
    const previous = content[index];
    return previous && signature(previous) === signature(item)
      ? { ...item, id: previous.id }
      : item;
  });

  const merged: TimelineItem[] = [];
  let cursor = 0;

  for (let index = 0; index <= reconciled.length; index += 1) {
    while (cursor < lifecycle.length && lifecycle[cursor].after <= index) {
      merged.push(lifecycle[cursor].item);
      cursor += 1;
    }
    if (index < reconciled.length) merged.push(reconciled[index]);
  }

  return merged;
}
