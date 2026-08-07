"use client";

import { useEffect, useState } from "react";

import {
  CloseIcon,
  DownloadIcon,
  ExternalIcon,
  FileIcon,
  SpinnerIcon,
} from "@/components/icons";
import { CopyButton } from "@/components/content/code-block";
import { runStatusToAgentStatus, StatusBadge } from "@/components/ui/status";
import { api } from "@/lib/api";
import {
  absoluteTime,
  bytes,
  cents,
  compactNumber,
  duration,
  exactNumber,
} from "@/lib/format";
import type {
  AgentSummary,
  AgentUsage,
  ArtifactEntry,
  RunSummary,
} from "@/lib/types";

function Row({
  label,
  value,
  mono,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className="shrink-0 text-[12px] text-ink-faint">{label}</span>
      <span
        className={
          mono
            ? "truncate font-mono text-[11.5px] text-ink-muted"
            : "truncate text-[12px] text-ink-muted"
        }
      >
        {value}
      </span>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-b border-hairline px-4 py-3.5 last:border-b-0">
      <h3 className="mb-2 text-[10px] font-semibold tracking-[0.08em] text-ink-faint uppercase">
        {title}
      </h3>
      {children}
    </section>
  );
}

export function DetailsPanel({
  agent,
  runs,
  onClose,
}: {
  agent: AgentSummary;
  runs: RunSummary[];
  onClose: () => void;
}) {
  const [usage, setUsage] = useState<AgentUsage | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const latest = runs.at(-1);
  const branches = latest?.git?.branches ?? [];

  // Cost settles a moment after a run ends, so this refetches while the last
  // run is still in flight.
  const running = latest?.status === "running";

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      const [usageResult, artifactResult] = await Promise.allSettled([
        api.usage(agent.agentId),
        api.artifacts(agent.agentId),
      ]);

      if (cancelled) return;

      if (usageResult.status === "fulfilled") setUsage(usageResult.value);
      if (artifactResult.status === "fulfilled") {
        setArtifacts(artifactResult.value.artifacts);
      }
      setLoading(false);
    };

    void load();

    if (!running) return;
    const interval = setInterval(() => void load(), 15_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [agent.agentId, running]);

  const tokens = usage?.usage ?? latest?.usage;

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col overflow-y-auto border-l border-hairline bg-canvas">
      <div className="flex items-center justify-between border-b border-hairline px-4 py-3">
        <h2 className="text-[13px] font-semibold text-ink">Run details</h2>
        <button
          type="button"
          aria-label="Close details"
          onClick={onClose}
          className="rounded p-1 text-ink-faint transition-colors hover:bg-raised hover:text-ink"
        >
          <CloseIcon className="size-4" />
        </button>
      </div>

      <Section title="Agent">
        <Row
          label="ID"
          mono
          value={
            <span className="inline-flex items-center gap-1">
              {agent.agentId.slice(0, 18)}…
              <CopyButton value={agent.agentId} label="Copy agent ID" />
            </span>
          }
        />
        <Row label="Created" value={absoluteTime(agent.createdAt) ?? "—"} />
        <Row label="Updated" value={absoluteTime(agent.lastModified) ?? "—"} />
        {agent.env ? (
          <Row
            label="Environment"
            value={agent.env.name ?? agent.env.type}
            mono
          />
        ) : null}
        {agent.repos.length > 0 ? (
          <Row label="Repositories" value={String(agent.repos.length)} />
        ) : null}
      </Section>

      {branches.length > 0 ? (
        <Section title="Git">
          {branches.map((branch, index) => (
            <div key={index} className="space-y-1 py-1">
              <Row label="Branch" mono value={branch.branch ?? "—"} />
              {branch.prUrl ? (
                <a
                  href={branch.prUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 text-[12px] text-accent hover:underline"
                >
                  View pull request
                  <ExternalIcon className="size-3" />
                </a>
              ) : null}
            </div>
          ))}
        </Section>
      ) : null}

      <Section title="Token usage">
        {loading && !tokens ? (
          <div className="flex items-center gap-2 py-1 text-[12px] text-ink-faint">
            <SpinnerIcon className="size-3.5 animate-spin-slow" />
            Loading
          </div>
        ) : tokens ? (
          <>
            <Row label="Input" value={exactNumber(tokens.inputTokens)} mono />
            <Row label="Output" value={exactNumber(tokens.outputTokens)} mono />
            <Row
              label="Cache read"
              value={exactNumber(tokens.cacheReadTokens)}
              mono
            />
            <Row
              label="Cache write"
              value={exactNumber(tokens.cacheWriteTokens)}
              mono
            />
            {tokens.reasoningTokens !== undefined ? (
              <Row
                label="Reasoning"
                value={exactNumber(tokens.reasoningTokens)}
                mono
              />
            ) : null}
            <div className="mt-1 border-t border-hairline pt-1">
              <Row
                label="Total"
                value={
                  <span className="text-ink">
                    {exactNumber(tokens.totalTokens)}
                  </span>
                }
                mono
              />
            </div>
          </>
        ) : (
          <p className="text-[12px] text-ink-faint">
            No usage reported for this agent yet.
          </p>
        )}

        {usage?.cost ? (
          <div className="mt-2 border-t border-hairline pt-2">
            <Row label="Charged" value={cents(usage.cost.chargedCents)} mono />
            <Row
              label="Raw model cost"
              value={cents(usage.cost.rawCostCents)}
              mono
            />
          </div>
        ) : null}
      </Section>

      <Section title={`Runs (${runs.length})`}>
        <ul className="space-y-2">
          {runs
            .slice()
            .reverse()
            .map((run) => (
              <li
                key={run.id}
                className="rounded-md border border-hairline bg-panel px-2.5 py-2"
              >
                <div className="flex items-center justify-between gap-2">
                  <StatusBadge status={runStatusToAgentStatus(run.status)} />
                  <span className="font-mono text-[11px] text-ink-faint">
                    {duration(run.durationMs)}
                  </span>
                </div>

                <div className="mt-1.5 space-y-0.5">
                  {run.model ? (
                    <p className="truncate font-mono text-[11px] text-ink-faint">
                      {run.model.id}
                      {run.model.params?.length
                        ? ` · ${run.model.params.map((p) => p.value).join(", ")}`
                        : ""}
                    </p>
                  ) : null}
                  {run.usage ? (
                    <p className="font-mono text-[11px] text-ink-faint">
                      {compactNumber(run.usage.totalTokens)} tokens
                    </p>
                  ) : null}
                  {run.requestId ? (
                    <p
                      className="truncate font-mono text-[10.5px] text-ink-faint/70"
                      title={run.requestId}
                    >
                      req {run.requestId}
                    </p>
                  ) : null}
                  {run.error ? (
                    <p className="text-[11px] text-negative">
                      {run.error.message}
                    </p>
                  ) : null}
                </div>
              </li>
            ))}
        </ul>
      </Section>

      {artifacts.length > 0 ? (
        <Section title={`Artifacts (${artifacts.length})`}>
          <ul className="space-y-1">
            {artifacts.map((artifact) => (
              <li key={artifact.path}>
                <a
                  href={api.artifactUrl(agent.agentId, artifact.path)}
                  className="group flex items-center gap-2 rounded-md px-1.5 py-1.5 transition-colors hover:bg-raised"
                >
                  <FileIcon className="size-3.5 shrink-0 text-ink-faint" />
                  <span className="min-w-0 flex-1 truncate text-[12px] text-ink-muted group-hover:text-ink">
                    {artifact.name}
                  </span>
                  <span className="shrink-0 font-mono text-[10.5px] text-ink-faint">
                    {bytes(artifact.sizeBytes)}
                  </span>
                  <DownloadIcon className="size-3.5 shrink-0 text-ink-faint opacity-0 transition-opacity group-hover:opacity-100" />
                </a>
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
    </aside>
  );
}
