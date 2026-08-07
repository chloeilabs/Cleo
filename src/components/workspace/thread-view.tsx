"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ArchiveIcon,
  BranchIcon,
  ChevronDownIcon,
  CopyIcon,
  ExternalIcon,
  InfoIcon,
  PullRequestIcon,
  RepoIcon,
  SpinnerIcon,
  TrashIcon,
} from "@/components/icons";
import { TimelineRow } from "@/components/timeline/timeline";
import { Button } from "@/components/ui/button";
import { Menu, MenuItem, MenuSeparator } from "@/components/ui/menu";
import {
  agentStatus,
  runStatusToAgentStatus,
  StatusBadge,
} from "@/components/ui/status";
import { useToast } from "@/components/ui/toast";
import {
  Composer,
  type ComposerSubmission,
} from "@/components/workspace/composer";
import { DetailsPanel } from "@/components/workspace/details-panel";
import { modelSelection, useWorkspace } from "@/components/workspace/provider";
import { useRunStream } from "@/hooks/use-run-stream";
import { api, type TranscriptRun } from "@/lib/api";
import {
  classNames,
  compactNumber,
  duration,
  firstLine,
  relativeTime,
} from "@/lib/format";
import type { AgentSummary, RunSummary, TimelineItem } from "@/lib/types";

interface ThreadState {
  agentId: string;
  agent: AgentSummary | null;
  runs: TranscriptRun[];
  /** The run currently being streamed, if any. */
  activeRunId?: string;
  error?: string;
}

/** Stable identity so `useMemo` consumers do not see a new array each render. */
const EMPTY_RUNS: TranscriptRun[] = [];

function upsertItem(items: TimelineItem[], item: TimelineItem): TimelineItem[] {
  const index = items.findIndex((entry) => entry.id === item.id);
  if (index === -1) return [...items, item];

  const next = [...items];
  next[index] = item;
  return next;
}

function RunFooter({ run }: { run: RunSummary }) {
  if (run.status === "running") return null;

  const parts = [
    run.model?.id,
    run.durationMs !== undefined ? duration(run.durationMs) : undefined,
    run.usage ? `${compactNumber(run.usage.totalTokens)} tokens` : undefined,
  ].filter(Boolean);

  if (parts.length === 0) return null;

  return (
    <p className="pt-0.5 pl-1 font-mono text-[11px] text-ink-faint">
      {parts.join(" · ")}
    </p>
  );
}

export function ThreadView({ agentId }: { agentId: string }) {
  const router = useRouter();
  const { reportError, notify } = useToast();
  const {
    preferences,
    selectedModel,
    refreshAgents,
    upsertAgent,
    removeAgent,
  } = useWorkspace();

  /**
   * The whole thread lives in one state object tagged with the agent it
   * belongs to, so switching agents swaps it atomically instead of resetting
   * four separate pieces of state on every navigation.
   */
  const [thread, setThread] = useState<ThreadState | null>(null);
  const [sending, setSending] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  const current = thread?.agentId === agentId ? thread : null;
  const loading = current === null;
  const agent = current?.agent ?? null;
  const runs = useMemo(() => current?.runs ?? EMPTY_RUNS, [current]);
  const loadError = current?.error;
  const activeRunId = current?.activeRunId;

  useEffect(() => {
    let cancelled = false;
    pinnedRef.current = true;

    void (async () => {
      try {
        const loaded = await api.transcript(agentId);
        if (cancelled) return;

        setThread({
          agentId,
          agent: loaded.agent,
          runs: loaded.runs,
          activeRunId: loaded.runs.find(
            (entry) => entry.run.status === "running",
          )?.run.id,
        });
      } catch (error) {
        if (cancelled) return;
        setThread({
          agentId,
          agent: null,
          runs: [],
          error:
            error instanceof Error
              ? error.message
              : "This agent could not load.",
        });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [agentId]);

  /** Apply an update to the thread, ignoring events for a superseded agent. */
  const patchThread = useCallback(
    (update: (state: ThreadState) => ThreadState) => {
      setThread((state) =>
        state && state.agentId === agentId ? update(state) : state,
      );
    },
    [agentId],
  );

  const patchRun = useCallback(
    (run: RunSummary) => {
      patchThread((state) => {
        const index = state.runs.findIndex((entry) => entry.run.id === run.id);
        if (index === -1) {
          return { ...state, runs: [...state.runs, { run, items: [] }] };
        }

        const runs = [...state.runs];
        runs[index] = { ...runs[index], run: { ...runs[index].run, ...run } };
        return { ...state, runs };
      });
    },
    [patchThread],
  );

  const reloadAgent = useCallback(async () => {
    try {
      const detail = await api.agent(agentId);
      patchThread((state) => ({ ...state, agent: detail.agent }));
      upsertAgent(detail.agent);
    } catch {
      // The sidebar poll will pick up the change instead.
    }
  }, [agentId, patchThread, upsertAgent]);

  useRunStream(agentId, activeRunId, {
    onRun: patchRun,

    onItem: (runId, item) => {
      patchThread((state) => {
        const index = state.runs.findIndex((entry) => entry.run.id === runId);
        if (index === -1) {
          return {
            ...state,
            runs: [
              ...state.runs,
              { run: { id: runId, agentId, status: "running" }, items: [item] },
            ],
          };
        }

        const runs = [...state.runs];
        runs[index] = {
          ...runs[index],
          items: upsertItem(runs[index].items, item),
        };
        return { ...state, runs };
      });
    },

    onTranscript: (runId, items) => {
      // The server's closing transcript is authoritative; adopt it wholesale so
      // a dropped connection can never leave a half-rendered turn behind.
      if (items.length === 0) return;

      patchThread((state) => {
        const index = state.runs.findIndex((entry) => entry.run.id === runId);
        if (index === -1) return state;

        const runs = [...state.runs];
        runs[index] = { ...runs[index], items };
        return { ...state, runs };
      });
    },

    onDone: () => {
      patchThread((state) => ({ ...state, activeRunId: undefined }));
      void reloadAgent();
      void refreshAgents();
    },

    onError: (error) => {
      patchThread((state) => ({ ...state, activeRunId: undefined }));
      notify({ tone: "error", title: error.error, helpUrl: error.helpUrl });
    },
  });

  // Follow new output only while the reader is already at the bottom.
  const items = useMemo(
    () => runs.flatMap((entry) => entry.items),
    [runs],
  );

  useEffect(() => {
    if (!pinnedRef.current) return;
    const node = scrollRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [items, loading]);

  const onScroll = () => {
    const node = scrollRef.current;
    if (!node) return;
    const distance = node.scrollHeight - node.scrollTop - node.clientHeight;
    pinnedRef.current = distance < 120;
  };

  const latestRun = runs.at(-1)?.run;
  const isRunning = latestRun?.status === "running" || Boolean(activeRunId);
  const branch = latestRun?.git?.branches?.[0];

  const send = async ({ prompt, images }: ComposerSubmission) => {
    setSending(true);
    pinnedRef.current = true;

    try {
      const { runId } = await api.sendMessage(agentId, {
        prompt,
        model: modelSelection(selectedModel),
        mode: preferences.mode,
        images: images.length > 0 ? images : undefined,
      });

      // Show the prompt immediately; the stream replaces it with Cursor's
      // stored copy as soon as the run reports back.
      patchThread((state) => ({
        ...state,
        activeRunId: runId,
        runs: [
          ...state.runs,
          {
            run: { id: runId, agentId, status: "running", createdAt: Date.now() },
            items: [{ id: `${runId}:prompt`, kind: "user", text: prompt }],
          },
        ],
      }));
      void refreshAgents();
    } catch (error) {
      reportError(error, "The message could not be sent.");
    } finally {
      setSending(false);
    }
  };

  const cancel = async () => {
    if (!activeRunId) return;
    try {
      await api.cancelRun(agentId, activeRunId);
      notify({ tone: "info", title: "Cancelling the run." });
    } catch (error) {
      reportError(error, "The run could not be cancelled.");
    }
  };

  const setArchived = async (archived: boolean) => {
    try {
      await api.setArchived(agentId, archived);
      notify({
        tone: "success",
        title: archived ? "Agent archived." : "Agent restored.",
      });
      await reloadAgent();
      void refreshAgents();
    } catch (error) {
      reportError(error);
    }
  };

  const remove = async () => {
    try {
      await api.deleteAgent(agentId);
      removeAgent(agentId);
      notify({ tone: "success", title: "Agent deleted." });
      router.push("/");
    } catch (error) {
      reportError(error);
    }
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center gap-2 text-[13px] text-ink-faint">
        <SpinnerIcon className="size-4 animate-spin-slow" />
        Loading conversation
      </div>
    );
  }

  if (loadError || !agent) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-[14px] text-ink">{loadError ?? "Agent not found."}</p>
        <Button onClick={() => router.push("/")}>Start a new agent</Button>
      </div>
    );
  }

  const title = agent.name?.trim() || firstLine(agent.summary || "Agent");

  return (
    <div className="flex h-full min-w-0">
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex shrink-0 items-center gap-3 border-b border-hairline px-4 py-2.5 pl-14 md:pl-4">
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-[14px] font-medium text-ink">
              {title}
            </h1>

            <div className="mt-0.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[11.5px] text-ink-faint">
              {agent.repos[0] ? (
                <a
                  href={agent.repos[0]}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 transition-colors hover:text-ink-muted"
                >
                  <RepoIcon className="size-3" />
                  {agent.repoLabel}
                </a>
              ) : (
                <span className="inline-flex items-center gap-1">
                  <RepoIcon className="size-3" />
                  Empty workspace
                </span>
              )}

              {branch?.branch ? (
                <span className="inline-flex items-center gap-1 font-mono">
                  <BranchIcon className="size-3" />
                  {branch.branch}
                </span>
              ) : null}

              <span>{relativeTime(agent.lastModified)}</span>
            </div>
          </div>

          <StatusBadge
            status={
              latestRun
                ? runStatusToAgentStatus(latestRun.status)
                : agentStatus(agent)
            }
          />

          {branch?.prUrl ? (
            <Button
              size="sm"
              onClick={() => window.open(branch.prUrl, "_blank")}
              className="hidden sm:inline-flex"
            >
              <PullRequestIcon className="size-3.5" />
              Pull request
            </Button>
          ) : null}

          <Button
            variant="ghost"
            size="icon"
            aria-label="Toggle run details"
            onClick={() => setShowDetails((value) => !value)}
            className={showDetails ? "bg-raised text-ink" : undefined}
          >
            <InfoIcon className="size-4" />
          </Button>

          <Menu
            label="Agent actions"
            align="end"
            triggerClassName="size-7 justify-center text-ink-faint hover:bg-raised hover:text-ink"
            trigger={<ChevronDownIcon className="size-4" />}
          >
            {(close) => (
              <>
                <MenuItem
                  onSelect={() => {
                    void navigator.clipboard.writeText(agent.agentId);
                    notify({ tone: "success", title: "Agent ID copied." });
                    close();
                  }}
                >
                  <span className="inline-flex items-center gap-2">
                    <CopyIcon className="size-3.5" />
                    Copy agent ID
                  </span>
                </MenuItem>

                <MenuItem
                  onSelect={() => {
                    window.open("https://cursor.com/agents", "_blank");
                    close();
                  }}
                >
                  <span className="inline-flex items-center gap-2">
                    <ExternalIcon className="size-3.5" />
                    Open in Cursor Web
                  </span>
                </MenuItem>

                <MenuSeparator />

                <MenuItem
                  onSelect={() => {
                    close();
                    void setArchived(!agent.archived);
                  }}
                >
                  <span className="inline-flex items-center gap-2">
                    <ArchiveIcon className="size-3.5" />
                    {agent.archived ? "Unarchive" : "Archive"}
                  </span>
                </MenuItem>

                <MenuItem
                  destructive
                  onSelect={() => {
                    close();
                    void remove();
                  }}
                >
                  <span className="inline-flex items-center gap-2">
                    <TrashIcon className="size-3.5" />
                    Delete agent
                  </span>
                </MenuItem>
              </>
            )}
          </Menu>
        </header>

        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="min-h-0 flex-1 overflow-y-auto"
        >
          <div className="mx-auto w-full max-w-3xl space-y-3 px-4 py-6">
            {runs.map((entry) => (
              <div key={entry.run.id} className="space-y-3">
                {entry.items.map((item) => (
                  <div key={item.id} className="animate-fade-up">
                    <TimelineRow item={item} />
                  </div>
                ))}
                <RunFooter run={entry.run} />
              </div>
            ))}

            {isRunning ? (
              <div className="flex items-center gap-2 pt-1 pl-1">
                <SpinnerIcon className="size-3.5 animate-spin-slow text-accent" />
                <span className="text-shimmer text-[13px] font-medium">
                  Working
                </span>
              </div>
            ) : null}

            {runs.length === 0 ? (
              <p className="py-10 text-center text-[13px] text-ink-faint">
                This agent has no messages yet.
              </p>
            ) : null}
          </div>
        </div>

        <div className="shrink-0 px-4 pb-4">
          <div className="mx-auto w-full max-w-3xl">
            <Composer
              variant="follow-up"
              busy={isRunning || sending}
              onCancel={activeRunId ? cancel : undefined}
              placeholder={
                isRunning
                  ? "Queue a follow-up once this run finishes…"
                  : "Send a follow-up. The agent keeps its full context."
              }
              onSubmit={send}
            />
          </div>
        </div>
      </div>

      {showDetails ? (
        <div className={classNames("hidden lg:block")}>
          <DetailsPanel
            agent={agent}
            runs={runs.map((entry) => entry.run)}
            onClose={() => setShowDetails(false)}
          />
        </div>
      ) : null}
    </div>
  );
}
