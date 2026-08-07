import { route } from "@/lib/server/errors";
import {
  currentSession,
  endSession,
  environmentApiKey,
  startSession,
} from "@/lib/server/session";
import type { SessionState } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export const GET = route(async () => {
  return Response.json(await currentSession());
});

export const POST = route(async (request: Request) => {
  const body = (await request.json()) as { apiKey?: string };
  const user = await startSession(body.apiKey ?? "");

  return Response.json({
    authenticated: true,
    fromEnvironment: false,
    user,
  } satisfies SessionState);
});

export const DELETE = route(async () => {
  await endSession();

  return Response.json({
    authenticated: Boolean(environmentApiKey()),
    fromEnvironment: Boolean(environmentApiKey()),
  } satisfies SessionState);
});
