import { getUsage } from "@/lib/server/cursor";
import { route } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ agentId: string }> };

export const GET = route(async (_request: Request, context: Context) => {
  const apiKey = await requireApiKey();
  const { agentId } = await context.params;

  return Response.json(await getUsage(apiKey, agentId));
});
