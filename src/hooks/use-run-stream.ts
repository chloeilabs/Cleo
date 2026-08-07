"use client";

import { useEffect, useRef } from "react";

import { api } from "@/lib/api";
import type { ApiError, RunSummary, TimelineItem } from "@/lib/types";

interface RunStreamHandlers {
  onRun: (run: RunSummary) => void;
  onItem: (runId: string, item: TimelineItem) => void;
  onTranscript: (runId: string, items: TimelineItem[]) => void;
  onDone: (runId: string) => void;
  onError: (error: ApiError) => void;
}

/**
 * Subscribe to one run's server-sent events.
 *
 * The server closes the stream once the run settles, so the connection is torn
 * down explicitly on `done` to stop `EventSource` from reconnecting in a loop.
 * Reopening later is always safe: the stream replays Cursor's stored transcript
 * before going live.
 */
export function useRunStream(
  agentId: string | undefined,
  runId: string | undefined,
  handlers: RunStreamHandlers,
): void {
  const handlersRef = useRef(handlers);

  useEffect(() => {
    handlersRef.current = handlers;
  });

  useEffect(() => {
    if (!agentId || !runId) return;

    const source = new EventSource(api.streamUrl(agentId, runId));
    let settled = false;

    const parse = <T,>(event: MessageEvent): T | undefined => {
      try {
        return JSON.parse(event.data) as T;
      } catch {
        return undefined;
      }
    };

    source.addEventListener("run", (event) => {
      const run = parse<RunSummary>(event as MessageEvent);
      if (run) handlersRef.current.onRun(run);
    });

    source.addEventListener("item", (event) => {
      const item = parse<TimelineItem>(event as MessageEvent);
      if (item) handlersRef.current.onItem(runId, item);
    });

    source.addEventListener("transcript", (event) => {
      const payload = parse<{ items: TimelineItem[] }>(event as MessageEvent);
      if (payload) handlersRef.current.onTranscript(runId, payload.items);
    });

    source.addEventListener("failed", (event) => {
      const payload = parse<ApiError>(event as MessageEvent);
      settled = true;
      handlersRef.current.onError(payload ?? { error: "The stream failed." });
      source.close();
    });

    source.addEventListener("done", () => {
      settled = true;
      handlersRef.current.onDone(runId);
      source.close();
    });

    source.onerror = () => {
      // A transport hiccup before the run settles is left to EventSource's own
      // retry. Anything after `done` means we already closed on purpose.
      if (settled) source.close();
    };

    return () => {
      settled = true;
      source.close();
    };
  }, [agentId, runId]);
}
