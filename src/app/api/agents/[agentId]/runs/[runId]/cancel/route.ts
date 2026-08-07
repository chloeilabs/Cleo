import { cancelRun } from "@/lib/server/cursor";
import { route } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ agentId: string; runId: string }> };

export const POST = route(async (_request: Request, context: Context) => {
  const apiKey = await requireApiKey();
  const { agentId, runId } = await context.params;

  await cancelRun(apiKey, agentId, runId);
  return Response.json({ ok: true });
});
