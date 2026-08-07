import type {
  AgentDetail,
  AgentSummary,
  AgentUsage,
  ApiError,
  ArtifactEntry,
  CreateAgentRequest,
  CreateAgentResponse,
  ModelOption,
  RepositoryOption,
  RunSummary,
  SendMessageRequest,
  SessionState,
  TimelineItem,
} from "@/lib/types";

/** Error carrying the structured payload the API routes return. */
export class RequestError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly helpUrl?: string;
  readonly retryable: boolean;

  constructor(status: number, payload: ApiError) {
    super(payload.error);
    this.name = "RequestError";
    this.status = status;
    this.code = payload.code;
    this.helpUrl = payload.helpUrl;
    this.retryable = payload.retryable ?? false;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit & { json?: unknown },
): Promise<T> {
  const { json, ...rest } = init ?? {};

  const response = await fetch(path, {
    ...rest,
    headers: {
      ...(json === undefined ? {} : { "Content-Type": "application/json" }),
      ...rest.headers,
    },
    ...(json === undefined ? {} : { body: JSON.stringify(json) }),
  });

  if (!response.ok) {
    const payload = (await response
      .json()
      .catch(() => ({ error: response.statusText }))) as ApiError;
    throw new RequestError(response.status, payload);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export interface TranscriptRun {
  run: RunSummary;
  items: TimelineItem[];
}

export const api = {
  session: () => request<SessionState>("/api/session"),

  signIn: (apiKey: string) =>
    request<SessionState>("/api/session", { method: "POST", json: { apiKey } }),

  signOut: () => request<SessionState>("/api/session", { method: "DELETE" }),

  models: () => request<{ models: ModelOption[] }>("/api/models"),

  repositories: () =>
    request<{ repositories: RepositoryOption[] }>("/api/repositories"),

  agents: (options: { includeArchived?: boolean; limit?: number } = {}) => {
    const params = new URLSearchParams();
    if (options.includeArchived) params.set("includeArchived", "true");
    if (options.limit) params.set("limit", String(options.limit));
    const query = params.toString();
    return request<{ items: AgentSummary[]; nextCursor?: string }>(
      `/api/agents${query ? `?${query}` : ""}`,
    );
  },

  agent: (agentId: string) =>
    request<AgentDetail>(`/api/agents/${encodeURIComponent(agentId)}`),

  transcript: (agentId: string) =>
    request<{ agent: AgentSummary; runs: TranscriptRun[] }>(
      `/api/agents/${encodeURIComponent(agentId)}/transcript`,
    ),

  createAgent: (body: CreateAgentRequest) =>
    request<CreateAgentResponse>("/api/agents", { method: "POST", json: body }),

  sendMessage: (agentId: string, body: SendMessageRequest) =>
    request<{ agentId: string; runId: string }>(
      `/api/agents/${encodeURIComponent(agentId)}/messages`,
      { method: "POST", json: body },
    ),

  cancelRun: (agentId: string, runId: string) =>
    request<{ ok: true }>(
      `/api/agents/${encodeURIComponent(agentId)}/runs/${encodeURIComponent(runId)}/cancel`,
      { method: "POST" },
    ),

  setArchived: (agentId: string, archived: boolean) =>
    request<{ ok: true }>(
      `/api/agents/${encodeURIComponent(agentId)}/archive`,
      { method: "POST", json: { archived } },
    ),

  deleteAgent: (agentId: string) =>
    request<{ ok: true }>(`/api/agents/${encodeURIComponent(agentId)}`, {
      method: "DELETE",
    }),

  usage: (agentId: string) =>
    request<AgentUsage>(`/api/agents/${encodeURIComponent(agentId)}/usage`),

  artifacts: (agentId: string) =>
    request<{ artifacts: ArtifactEntry[] }>(
      `/api/agents/${encodeURIComponent(agentId)}/artifacts`,
    ),

  artifactUrl: (agentId: string, path: string) =>
    `/api/agents/${encodeURIComponent(agentId)}/artifacts/download?path=${encodeURIComponent(path)}`,

  streamUrl: (agentId: string, runId: string) =>
    `/api/agents/${encodeURIComponent(agentId)}/runs/${encodeURIComponent(runId)}/stream`,
};
