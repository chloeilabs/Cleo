import { listRepositories } from "@/lib/server/cursor";
import { route } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export const GET = route(async () => {
  const apiKey = await requireApiKey();
  return Response.json({ repositories: await listRepositories(apiKey) });
});
