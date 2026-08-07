import "server-only";

import { createHash } from "node:crypto";

import { Agent, Cursor } from "@cursor/sdk";
import type {
  ModelSelection,
  Run,
  SDKAgent,
  SDKAgentInfo,
  SDKUserMessage,
} from "@cursor/sdk";

import type {
  AgentSummary,
  ArtifactEntry,
  CreateAgentRequest,
  ModelOption,
  RepositoryOption,
  RunSummary,
  SendMessageRequest,
} from "@/lib/types";

/**
 * Thin, typed wrapper over `@cursor/sdk` for the cloud runtime.
 *
 * Cursor already persists agents, runs, and transcripts server-side, so this
 * app deliberately keeps no database of its own — every read goes back to the
 * SDK and Cursor stays the single source of truth.
 */

const AGENT_CACHE_LIMIT = 16;
const AGENT_CACHE_TTL_MS = 15 * 60 * 1000;

interface CachedAgent {
  agent: SDKAgent;
  touchedAt: number;
}

const globalCache = globalThis as typeof globalThis & {
  __cleoAgentCache?: Map<string, CachedAgent>;
};

const agentCache = (globalCache.__cleoAgentCache ??= new Map<
  string,
  CachedAgent
>());

function cacheKey(apiKey: string, agentId: string): string {
  const fingerprint = createHash("sha256")
    .update(apiKey)
    .digest("hex")
    .slice(0, 16);
  return `${fingerprint}:${agentId}`;
}

function release(entry: CachedAgent): void {
  try {
    entry.agent.close();
  } catch {
    // Disposal is best-effort; a cloud run keeps going without the handle.
  }
}

function sweep(): void {
  const now = Date.now();
  for (const [key, entry] of agentCache) {
    if (now - entry.touchedAt > AGENT_CACHE_TTL_MS) {
      agentCache.delete(key);
      release(entry);
    }
  }
}

function remember(apiKey: string, agent: SDKAgent): void {
  sweep();

  const key = cacheKey(apiKey, agent.agentId);
  agentCache.delete(key);
  agentCache.set(key, { agent, touchedAt: Date.now() });

  while (agentCache.size > AGENT_CACHE_LIMIT) {
    const oldest = agentCache.keys().next();
    if (oldest.done) break;
    const evicted = agentCache.get(oldest.value);
    agentCache.delete(oldest.value);
    if (evicted) release(evicted);
  }
}

export function forgetAgent(apiKey: string, agentId: string): void {
  const key = cacheKey(apiKey, agentId);
  const entry = agentCache.get(key);
  if (!entry) return;
  agentCache.delete(key);
  release(entry);
}

/**
 * Reattach to a cloud agent, reusing the in-process handle when one is warm.
 * A cold instance simply resumes again — the cache is an optimisation, never a
 * correctness requirement.
 */
async function attach(apiKey: string, agentId: string): Promise<SDKAgent> {
  const key = cacheKey(apiKey, agentId);
  const cached = agentCache.get(key);

  if (cached && Date.now() - cached.touchedAt <= AGENT_CACHE_TTL_MS) {
    cached.touchedAt = Date.now();
    return cached.agent;
  }

  if (cached) {
    agentCache.delete(key);
    release(cached);
  }

  const agent = await Agent.resume(agentId, { apiKey });
  remember(apiKey, agent);
  return agent;
}

export function repoLabel(url: string): string {
  return url
    .trim()
    .replace(/\.git$/, "")
    .replace(/^https?:\/\/(www\.)?(github|gitlab|bitbucket)\.com\//, "")
    .replace(/^git@[^:]+:/, "")
    .replace(/^https?:\/\//, "");
}

function toAgentSummary(info: SDKAgentInfo): AgentSummary {
  const repos = info.runtime === "cloud" ? (info.repos ?? []) : [];

  return {
    agentId: info.agentId,
    name: info.name,
    summary: info.summary,
    status: info.status,
    archived: info.archived ?? false,
    createdAt: info.createdAt,
    lastModified: info.lastModified,
    repos,
    repoLabel: repos[0] ? repoLabel(repos[0]) : undefined,
    env: info.runtime === "cloud" ? info.env : undefined,
  };
}

export function toRunSummary(run: Run): RunSummary {
  return {
    id: run.id,
    agentId: run.agentId,
    requestId: run.requestId,
    status: run.status,
    result: run.result,
    error: run.error,
    model: run.model,
    durationMs: run.durationMs,
    usage: run.usage,
    git: run.git,
    createdAt: run.createdAt,
  };
}

export async function listAgents(
  apiKey: string,
  options: { cursor?: string; includeArchived?: boolean; limit?: number } = {},
): Promise<{ items: AgentSummary[]; nextCursor?: string }> {
  const result = await Agent.list({
    runtime: "cloud",
    apiKey,
    limit: options.limit ?? 50,
    cursor: options.cursor,
    includeArchived: options.includeArchived ?? false,
  });

  return {
    items: result.items.map(toAgentSummary),
    nextCursor: result.nextCursor,
  };
}

export async function getAgent(
  apiKey: string,
  agentId: string,
): Promise<AgentSummary> {
  return toAgentSummary(await Agent.get(agentId, { apiKey }));
}

/** Runs for an agent, oldest first so the thread reads top to bottom. */
export async function listRuns(
  apiKey: string,
  agentId: string,
  limit = 20,
): Promise<Run[]> {
  const result = await Agent.listRuns(agentId, {
    runtime: "cloud",
    apiKey,
    limit,
  });

  return [...result.items].sort((a, b) => {
    const left = a.createdAt ?? 0;
    const right = b.createdAt ?? 0;
    if (left !== right) return left - right;
    return 0;
  });
}

export async function getRun(
  apiKey: string,
  agentId: string,
  runId: string,
): Promise<Run> {
  return Agent.getRun(runId, { runtime: "cloud", agentId, apiKey });
}

function toUserMessage(
  prompt: string,
  images?: Array<{ data: string; mimeType: string }>,
): string | SDKUserMessage {
  if (!images?.length) return prompt;
  return { text: prompt, images };
}

export async function createAgent(
  apiKey: string,
  input: CreateAgentRequest,
): Promise<{ agentId: string; runId: string }> {
  const prompt = input.prompt.trim();
  if (!prompt) throw new Error("Write a prompt before starting an agent.");

  const startingRef = input.startingRef?.trim();
  const repos = input.repoUrl?.trim()
    ? [{ url: input.repoUrl.trim(), ...(startingRef ? { startingRef } : {}) }]
    : undefined;

  const agent = await Agent.create({
    apiKey,
    ...(input.model ? { model: input.model } : {}),
    ...(input.mode ? { mode: input.mode } : {}),
    ...(input.name?.trim() ? { name: input.name.trim() } : {}),
    cloud: {
      ...(repos ? { repos } : {}),
      autoCreatePR: input.autoCreatePR ?? false,
      workOnCurrentBranch: input.workOnCurrentBranch ?? false,
    },
  });

  remember(apiKey, agent);

  const run = await agent.send(toUserMessage(prompt, input.images));
  return { agentId: agent.agentId, runId: run.id };
}

export async function sendMessage(
  apiKey: string,
  agentId: string,
  input: SendMessageRequest,
): Promise<{ runId: string }> {
  const prompt = input.prompt.trim();
  if (!prompt) throw new Error("Write a message before sending.");

  const agent = await attach(apiKey, agentId);
  const run = await agent.send(toUserMessage(prompt, input.images), {
    ...(input.model ? { model: input.model } : {}),
    ...(input.mode ? { mode: input.mode } : {}),
  });

  return { runId: run.id };
}

export async function cancelRun(
  apiKey: string,
  agentId: string,
  runId: string,
): Promise<void> {
  await Agent.cancelRun(runId, { runtime: "cloud", agentId, apiKey });
}

export async function archiveAgent(
  apiKey: string,
  agentId: string,
  archived: boolean,
): Promise<void> {
  if (archived) {
    await Agent.archive(agentId, { apiKey });
  } else {
    await Agent.unarchive(agentId, { apiKey });
  }
}

export async function deleteAgent(
  apiKey: string,
  agentId: string,
): Promise<void> {
  forgetAgent(apiKey, agentId);
  await Agent.delete(agentId, { apiKey });
}

export async function getUsage(apiKey: string, agentId: string) {
  return Agent.getUsage(agentId, { apiKey });
}

export async function listArtifacts(
  apiKey: string,
  agentId: string,
): Promise<ArtifactEntry[]> {
  const agent = await attach(apiKey, agentId);
  const artifacts = await agent.listArtifacts();

  return artifacts.map((artifact) => ({
    path: artifact.path,
    name: artifact.path.split("/").filter(Boolean).at(-1) ?? artifact.path,
    sizeBytes: artifact.sizeBytes,
    updatedAt: artifact.updatedAt,
  }));
}

export async function downloadArtifact(
  apiKey: string,
  agentId: string,
  path: string,
): Promise<Buffer> {
  const agent = await attach(apiKey, agentId);
  return agent.downloadArtifact(path);
}

/**
 * Flatten the model catalog into dropdown entries: one per model, plus one per
 * preset variant so parameterised models (Router's Cost/Balance/Intelligence,
 * Composer's Fast) are directly selectable.
 */
export async function listModels(apiKey: string): Promise<ModelOption[]> {
  const models = await Cursor.models.list({ apiKey });
  const options: ModelOption[] = [];

  for (const model of models) {
    const variants = model.variants ?? [];

    if (variants.length === 0) {
      options.push({
        key: model.id,
        modelId: model.id,
        label: model.displayName || model.id,
        description: model.description,
        parameters: model.parameters,
      });
      continue;
    }

    for (const variant of variants) {
      const suffix = variant.params
        .map((param) => `${param.id}=${param.value}`)
        .join(",");

      options.push({
        key: `${model.id}::${suffix}`,
        modelId: model.id,
        label: variant.displayName || model.displayName || model.id,
        description: variant.description ?? model.description,
        params: variant.params,
        isDefault: variant.isDefault,
        parameters: model.parameters,
      });
    }
  }

  return options;
}

export async function listRepositories(
  apiKey: string,
): Promise<RepositoryOption[]> {
  const repositories = await Cursor.repositories.list({ apiKey });

  return repositories
    .map((repository) => {
      const label = repoLabel(repository.url);
      const [owner, name] = label.split("/");
      return { url: repository.url, label, owner, name };
    })
    .sort((a, b) => a.label.localeCompare(b.label));
}

export type { ModelSelection };
