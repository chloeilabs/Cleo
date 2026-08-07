import { listArtifacts } from "@/lib/server/cursor";
import { route } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ agentId: string }> };

export const GET = route(async (_request: Request, context: Context) => {
  const apiKey = await requireApiKey();
  const { agentId } = await context.params;

  try {
    return Response.json({ artifacts: await listArtifacts(apiKey, agentId) });
  } catch {
    // Artifact support is runtime dependent; an agent without it is not an
    // error, it just has nothing to show.
    return Response.json({ artifacts: [] });
  }
});
