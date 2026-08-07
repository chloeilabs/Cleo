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
