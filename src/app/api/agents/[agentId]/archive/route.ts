import { archiveAgent } from "@/lib/server/cursor";
import { route } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ agentId: string }> };

export const POST = route(async (request: Request, context: Context) => {
  const apiKey = await requireApiKey();
  const { agentId } = await context.params;
  const body = (await request.json().catch(() => ({}))) as {
    archived?: boolean;
  };

  await archiveAgent(apiKey, agentId, body.archived ?? true);
  return Response.json({ ok: true });
});
