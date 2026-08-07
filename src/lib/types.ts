/**
 * Wire types shared by the API routes and the browser.
 *
 * Everything the client renders is normalized into these shapes on the server
 * so that a live SSE stream and a replayed transcript produce identical UI.
 */

export type RunStatus = "running" | "finished" | "error" | "cancelled";

export type CloudStatus =
  | "CREATING"
  | "RUNNING"
  | "FINISHED"
  | "ERROR"
  | "CANCELLED"
  | "EXPIRED";

export type ConversationMode = "agent" | "plan";

export type ToolStatus = "running" | "completed" | "error";

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  totalTokens: number;
  reasoningTokens?: number;
}

export interface UsageCost {
  rawCostCents: number;
  chargedCents: number;
}

export interface AgentUsage {
  usage: TokenUsage;
  cost?: UsageCost;
  runs: Array<{ runId: string; usage: TokenUsage; cost?: UsageCost }>;
}

export interface GitBranch {
  repoUrl: string;
  branch?: string;
  prUrl?: string;
}

/** A todo entry emitted by the agent's `updateTodos` tool. */
export interface TodoEntry {
  content: string;
  status: "pending" | "inProgress" | "completed" | "cancelled";
}

/**
 * The presentational payload for a tool call. Tool `args`/`result` schemas are
 * explicitly unstable in the SDK, so the server parses them defensively and
 * hands the client a small, stable projection plus the raw JSON for the
 * "expand" affordance.
 */
export interface ToolView {
  /** Tool name as reported by the SDK (`shell`, `edit`, `read`, ...). */
  name: string;
  status: ToolStatus;
  /** Short headline, e.g. the shell command or the edited file path. */
  title: string;
  /** Secondary context, e.g. `+12 -3` for an edit or an exit code. */
  subtitle?: string;
  /** Monospaced body: command output, file preview, search results. */
  body?: string;
  /** Unified diff text when the tool produced one. */
  diff?: string;
  /** Todo list snapshot for `updateTodos`. */
  todos?: TodoEntry[];
  /** Markdown plan for `createPlan`. */
  markdown?: string;
  /** Pretty-printed arguments, for the raw view. */
  rawArgs?: string;
  /** Pretty-printed result, for the raw view. */
  rawResult?: string;
  errorText?: string;
  truncated?: boolean;
}

export type TimelineItem =
  | { id: string; kind: "user"; text: string }
  | { id: string; kind: "assistant"; text: string }
  | { id: string; kind: "thinking"; text: string; durationMs?: number }
  | { id: string; kind: "tool"; callId: string; tool: ToolView }
  | {
      id: string;
      kind: "shell";
      command: string;
      workingDirectory?: string;
      stdout?: string;
      stderr?: string;
      exitCode?: number;
    }
  | { id: string; kind: "status"; status: CloudStatus; message?: string }
  | { id: string; kind: "usage"; usage: TokenUsage }
  | { id: string; kind: "error"; message: string; code?: string };

export interface RunSummary {
  id: string;
  agentId: string;
  requestId?: string;
  status: RunStatus;
  result?: string;
  error?: { message: string; code?: string };
  model?: { id: string; params?: Array<{ id: string; value: string }> };
  durationMs?: number;
  usage?: TokenUsage;
  git?: { branches: GitBranch[] };
  createdAt?: number;
}

export interface AgentSummary {
  agentId: string;
  name: string;
  summary: string;
  status?: "running" | "finished" | "error";
  archived: boolean;
  createdAt?: number;
  lastModified: number;
  repos: string[];
  /** `owner/name` label derived from the first repo. */
  repoLabel?: string;
  env?: { type: string; name?: string };
}

export interface AgentDetail {
  agent: AgentSummary;
  runs: RunSummary[];
}

export interface ModelParameterOption {
  value: string;
  displayName?: string;
}

export interface ModelParameter {
  id: string;
  displayName?: string;
  values: ModelParameterOption[];
}

/** One selectable entry in the model dropdown (a model, or one of its variants). */
export interface ModelOption {
  /** Stable key for the dropdown, e.g. `composer-2.5` or `auto-smart::balanced`. */
  key: string;
  modelId: string;
  label: string;
  description?: string;
  params?: Array<{ id: string; value: string }>;
  isDefault?: boolean;
  parameters?: ModelParameter[];
}

export interface RepositoryOption {
  url: string;
  label: string;
  owner?: string;
  name?: string;
}

export interface SessionUser {
  apiKeyName: string;
  userId?: number;
  email?: string;
  firstName?: string;
  lastName?: string;
  displayName: string;
  createdAt: string;
}

export interface SessionState {
  authenticated: boolean;
  /** True when the key comes from `CURSOR_API_KEY` and cannot be signed out. */
  fromEnvironment: boolean;
  user?: SessionUser;
}

export interface ArtifactEntry {
  path: string;
  name: string;
  sizeBytes: number;
  updatedAt: string;
}

export interface CreateAgentRequest {
  prompt: string;
  repoUrl?: string;
  startingRef?: string;
  model?: { id: string; params?: Array<{ id: string; value: string }> };
  mode?: ConversationMode;
  autoCreatePR?: boolean;
  workOnCurrentBranch?: boolean;
  name?: string;
  images?: Array<{ data: string; mimeType: string }>;
}

export interface SendMessageRequest {
  prompt: string;
  model?: { id: string; params?: Array<{ id: string; value: string }> };
  mode?: ConversationMode;
  images?: Array<{ data: string; mimeType: string }>;
}

export interface CreateAgentResponse {
  agentId: string;
  runId: string;
}

export interface ApiError {
  error: string;
  code?: string;
  helpUrl?: string;
  retryable?: boolean;
}

/** Payload of the `snapshot` SSE event, sent once when a stream is opened. */
export interface StreamSnapshot {
  run: RunSummary;
  items: TimelineItem[];
}
