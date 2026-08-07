import { sendMessage } from "@/lib/server/cursor";
import { route } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";
import type { SendMessageRequest } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ agentId: string }> };

export const POST = route(async (request: Request, context: Context) => {
  const apiKey = await requireApiKey();
  const { agentId } = await context.params;
  const body = (await request.json()) as SendMessageRequest;

  const { runId } = await sendMessage(apiKey, agentId, body);
  return Response.json({ agentId, runId }, { status: 201 });
});
