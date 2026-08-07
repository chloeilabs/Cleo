import { getRun, toRunSummary } from "@/lib/server/cursor";
import { mapError } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";
import { itemFromStreamEvent, itemsFromConversation } from "@/lib/server/timeline";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const HEARTBEAT_MS = 15_000;

type Context = { params: Promise<{ agentId: string; runId: string }> };

/**
 * Server-sent events for one run.
 *
 * Live runs are streamed from `run.stream()`; settled runs replay
 * `run.conversation()`. Either way the stream closes with an authoritative
 * `transcript` event so the client's timeline matches what Cursor stored, even
 * if the connection dropped and reconnected midway.
 */
export async function GET(request: Request, context: Context) {
  let apiKey: string;
  let agentId: string;
  let runId: string;

  try {
    apiKey = await requireApiKey();
    ({ agentId, runId } = await context.params);
  } catch (error) {
    const { status, body } = mapError(error);
    return Response.json(body, { status });
  }

  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let closed = false;

      const send = (event: string, data: unknown) => {
        if (closed) return;
        try {
          controller.enqueue(
            encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
          );
        } catch {
          closed = true;
        }
      };

      const heartbeat = setInterval(() => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(": keep-alive\n\n"));
        } catch {
          closed = true;
        }
      }, HEARTBEAT_MS);

      const finish = () => {
        clearInterval(heartbeat);
        if (closed) return;
        closed = true;
        try {
          controller.close();
        } catch {
          // The consumer already went away.
        }
      };

      request.signal.addEventListener("abort", finish);

      try {
        const run = await getRun(apiKey, agentId, runId);
        send("run", toRunSummary(run));

        const unsubscribe = run.onDidChangeStatus(() => {
          send("run", toRunSummary(run));
        });

        try {
          if (run.status === "running" && run.supports("stream")) {
            let sequence = 0;
            for await (const event of run.stream()) {
              if (closed || request.signal.aborted) break;
              const item = itemFromStreamEvent(event, sequence++);
              if (item) send("item", item);
            }
          }

          if (!closed && !request.signal.aborted) {
            if (run.supports("conversation")) {
              send("transcript", {
                items: itemsFromConversation(await run.conversation()),
              });
            }
            send("run", toRunSummary(run));
          }
        } finally {
          unsubscribe();
        }

        send("done", { runId });
      } catch (error) {
        send("failed", mapError(error).body);
      } finally {
        request.signal.removeEventListener("abort", finish);
        finish();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
