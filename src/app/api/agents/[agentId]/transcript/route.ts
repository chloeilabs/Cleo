import { getAgent, listRuns, toRunSummary } from "@/lib/server/cursor";
import { route } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";
import { itemsFromConversation } from "@/lib/server/timeline";
import type { AgentSummary, RunSummary, TimelineItem } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ agentId: string }> };

export interface TranscriptResponse {
  agent: AgentSummary;
  runs: Array<{ run: RunSummary; items: TimelineItem[] }>;
}

/**
 * The full thread for an agent: every run in order, each with its transcript.
 * Cursor stores all of this server-side, so a reload rebuilds the exact same
 * conversation without any local persistence.
 */
export const GET = route(async (_request: Request, context: Context) => {
  const apiKey = await requireApiKey();
  const { agentId } = await context.params;

  const [agent, runs] = await Promise.all([
    getAgent(apiKey, agentId),
    listRuns(apiKey, agentId),
  ]);

  const transcripts = await Promise.all(
    runs.map(async (run) => {
      const summary = toRunSummary(run);

      if (!run.supports("conversation")) {
        return { run: summary, items: [] as TimelineItem[] };
      }

      try {
        return {
          run: summary,
          items: itemsFromConversation(await run.conversation()),
        };
      } catch {
        // A run whose transcript is not readable yet still belongs in the
        // thread — render its metadata and let the live stream fill it in.
        return { run: summary, items: [] as TimelineItem[] };
      }
    }),
  );

  return Response.json({
    agent,
    runs: transcripts,
  } satisfies TranscriptResponse);
});
