import {
  deleteAgent,
  getAgent,
  listRuns,
  toRunSummary,
} from "@/lib/server/cursor";
import { route } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";
import type { AgentDetail } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ agentId: string }> };

export const GET = route(async (_request: Request, context: Context) => {
  const apiKey = await requireApiKey();
  const { agentId } = await context.params;

  const [agent, runs] = await Promise.all([
    getAgent(apiKey, agentId),
    listRuns(apiKey, agentId),
  ]);

  return Response.json({
    agent,
    runs: runs.map(toRunSummary),
  } satisfies AgentDetail);
});

export const DELETE = route(async (_request: Request, context: Context) => {
  const apiKey = await requireApiKey();
  const { agentId } = await context.params;

  await deleteAgent(apiKey, agentId);
  return Response.json({ ok: true });
});
