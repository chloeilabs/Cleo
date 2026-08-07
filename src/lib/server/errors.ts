import "server-only";

import {
  AgentBusyError,
  AgentNotFoundError,
  AuthenticationError,
  ConfigurationError,
  CursorSdkError,
  IntegrationNotConnectedError,
  NetworkError,
  RateLimitError,
} from "@cursor/sdk";

import { InvalidApiKeyError, MissingApiKeyError } from "@/lib/server/session";
import type { ApiError } from "@/lib/types";

interface Mapped {
  status: number;
  body: ApiError;
}

/** Translate an SDK or session error into the shape the client renders. */
export function mapError(error: unknown): Mapped {
  if (error instanceof MissingApiKeyError) {
    return { status: 401, body: { error: error.message, code: error.code } };
  }

  if (error instanceof InvalidApiKeyError) {
    return { status: 401, body: { error: error.message, code: error.code } };
  }

  if (error instanceof IntegrationNotConnectedError) {
    return {
      status: 400,
      body: {
        error: `${error.provider} is not connected to your Cursor team. Connect it, then try again.`,
        code: "integration_not_connected",
        helpUrl: error.helpUrl,
      },
    };
  }

  if (error instanceof AgentBusyError) {
    return {
      status: 409,
      body: {
        error:
          "This agent is still working on the previous message. Wait for it to finish or cancel the active run.",
        code: "agent_busy",
      },
    };
  }

  if (error instanceof AuthenticationError) {
    return {
      status: 401,
      body: {
        error:
          "Cursor rejected the API key. Create a new one at cursor.com/dashboard/api.",
        code: "invalid_api_key",
      },
    };
  }

  if (error instanceof RateLimitError) {
    return {
      status: 429,
      body: { error: error.message, code: "rate_limited", retryable: true },
    };
  }

  if (error instanceof AgentNotFoundError) {
    return { status: 404, body: { error: error.message, code: "not_found" } };
  }

  if (error instanceof ConfigurationError) {
    return { status: 400, body: { error: error.message, code: error.code } };
  }

  if (error instanceof NetworkError) {
    return {
      status: 503,
      body: { error: error.message, code: error.code, retryable: true },
    };
  }

  if (error instanceof CursorSdkError) {
    return {
      status: error.status ?? 500,
      body: {
        error: error.message,
        code: error.code,
        retryable: error.isRetryable,
      },
    };
  }

  return {
    status: 500,
    body: {
      error: error instanceof Error ? error.message : "Something went wrong.",
    },
  };
}

export function errorResponse(error: unknown): Response {
  const { status, body } = mapError(error);
  return Response.json(body, { status });
}

/** Wrap a route handler so every thrown error becomes a structured response. */
export function route<A extends unknown[]>(
  handler: (...args: A) => Promise<Response>,
): (...args: A) => Promise<Response> {
  return async (...args: A) => {
    try {
      return await handler(...args);
    } catch (error) {
      return errorResponse(error);
    }
  };
}
