import { downloadArtifact } from "@/lib/server/cursor";
import { route } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ agentId: string }> };

export const GET = route(async (request: Request, context: Context) => {
  const apiKey = await requireApiKey();
  const { agentId } = await context.params;
  const path = new URL(request.url).searchParams.get("path");

  if (!path) {
    return Response.json({ error: "A `path` query parameter is required." }, {
      status: 400,
    });
  }

  const buffer = await downloadArtifact(apiKey, agentId, path);
  const name = path.split("/").filter(Boolean).at(-1) ?? "artifact";

  return new Response(new Uint8Array(buffer), {
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Disposition": `attachment; filename="${name.replace(/"/g, "")}"`,
      "Content-Length": String(buffer.byteLength),
    },
  });
});
