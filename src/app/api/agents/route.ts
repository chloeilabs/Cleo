import { createAgent, listAgents } from "@/lib/server/cursor";
import { route } from "@/lib/server/errors";
import { requireApiKey } from "@/lib/server/session";
import type { CreateAgentRequest, CreateAgentResponse } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export const GET = route(async (request: Request) => {
  const apiKey = await requireApiKey();
  const params = new URL(request.url).searchParams;

  const result = await listAgents(apiKey, {
    cursor: params.get("cursor") ?? undefined,
    includeArchived: params.get("includeArchived") === "true",
    limit: Number(params.get("limit")) || undefined,
  });

  return Response.json(result);
});

export const POST = route(async (request: Request) => {
  const apiKey = await requireApiKey();
  const body = (await request.json()) as CreateAgentRequest;

  const created = await createAgent(apiKey, body);
  return Response.json(created satisfies CreateAgentResponse, { status: 201 });
});
