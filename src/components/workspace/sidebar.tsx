"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  ArchiveIcon,
  KeyIcon,
  LogoIcon,
  PlusIcon,
  RepoIcon,
  SearchIcon,
  SpinnerIcon,
} from "@/components/icons";
import { Button } from "@/components/ui/button";
import { Menu, MenuItem, MenuLabel, MenuSeparator } from "@/components/ui/menu";
import { agentStatus, StatusDot } from "@/components/ui/status";
import { useToast } from "@/components/ui/toast";
import { useWorkspace } from "@/components/workspace/provider";
import { api } from "@/lib/api";
import { classNames, firstLine, relativeTime } from "@/lib/format";
import type { AgentSummary } from "@/lib/types";

function AgentRow({
  agent,
  active,
}: {
  agent: AgentSummary;
  active: boolean;
}) {
  const status = agentStatus(agent);
  const title = agent.name?.trim() || firstLine(agent.summary || "New agent");

  return (
    <Link
      href={`/agents/${encodeURIComponent(agent.agentId)}`}
      prefetch={false}
      className={classNames(
        "group block rounded-lg border px-2.5 py-2 transition-colors",
        active
          ? "border-hairline-strong bg-raised"
          : "border-transparent hover:border-hairline hover:bg-panel",
      )}
    >
      <div className="flex items-start gap-2">
        <span className="mt-1.5">
          <StatusDot status={status} />
        </span>

        <div className="min-w-0 flex-1">
          <p
            className={classNames(
              "truncate text-[13px] leading-snug",
              active ? "text-ink" : "text-ink-muted group-hover:text-ink",
            )}
          >
            {title}
          </p>

          <div className="mt-1 flex items-center gap-1.5 text-[11px] text-ink-faint">
            {agent.repoLabel ? (
              <>
                <RepoIcon className="size-3 shrink-0" />
                <span className="truncate">{agent.repoLabel}</span>
                <span aria-hidden>·</span>
              </>
            ) : null}
            <span className="shrink-0">{relativeTime(agent.lastModified)}</span>
          </div>
        </div>
      </div>
    </Link>
  );
}

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const router = useRouter();
  const params = useParams<{ agentId?: string }>();
  const { reportError, notify } = useToast();
  const {
    session,
    agents,
    agentsLoading,
    showArchived,
    setShowArchived,
    refreshAgents,
  } = useWorkspace();

  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);

  const activeAgentId = params?.agentId
    ? decodeURIComponent(params.agentId)
    : undefined;

  // `/` focuses search, the way it does in Cursor's agent list.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable;

      if (event.key === "/" && !typing) {
        event.preventDefault();
        searchRef.current?.focus();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return agents;

    return agents.filter((agent) =>
      [agent.name, agent.summary, agent.repoLabel, agent.agentId]
        .filter(Boolean)
        .some((field) => field!.toLowerCase().includes(needle)),
    );
  }, [agents, query]);

  const { running, rest } = useMemo(
    () => ({
      running: filtered.filter((agent) => agent.status === "running"),
      rest: filtered.filter((agent) => agent.status !== "running"),
    }),
    [filtered],
  );

  const signOut = async () => {
    try {
      const next = await api.signOut();
      if (next.authenticated) {
        notify({
          tone: "info",
          title: "Still signed in through CURSOR_API_KEY.",
          detail: "Remove the environment variable to sign out completely.",
        });
      }
      router.refresh();
    } catch (error) {
      reportError(error);
    }
  };

  return (
    <div className="flex h-full flex-col bg-canvas">
      <div className="flex items-center gap-2 px-3 pt-3 pb-2">
        <LogoIcon className="size-6 shrink-0 text-accent" />
        <span className="flex-1 text-[13px] font-semibold tracking-tight text-ink">
          Cleo
        </span>

        <Menu
          label="Account"
          align="end"
          triggerClassName="size-7 justify-center text-ink-faint hover:bg-raised hover:text-ink"
          trigger={
            <span className="flex size-6 items-center justify-center rounded-full bg-raised text-[10px] font-semibold text-ink-muted">
              {(session.user?.displayName ?? "?").slice(0, 1).toUpperCase()}
            </span>
          }
        >
          {(close) => (
            <>
              <MenuLabel>Signed in</MenuLabel>
              <div className="px-2.5 pb-2">
                <p className="truncate text-[13px] text-ink">
                  {session.user?.displayName ?? "Cursor user"}
                </p>
                <p className="truncate text-[11px] text-ink-faint">
                  {session.user?.email ?? session.user?.apiKeyName}
                </p>
              </div>

              <MenuSeparator />

              <MenuItem
                onSelect={() => {
                  window.open("https://cursor.com/dashboard/usage", "_blank");
                  close();
                }}
              >
                Usage dashboard
              </MenuItem>
              <MenuItem
                onSelect={() => {
                  window.open("https://cursor.com/agents", "_blank");
                  close();
                }}
              >
                Open in Cursor Web
              </MenuItem>

              <MenuSeparator />

              <MenuItem
                destructive
                onSelect={() => {
                  close();
                  void signOut();
                }}
              >
                Sign out
              </MenuItem>
            </>
          )}
        </Menu>
      </div>

      <div className="px-3 pb-2">
        <Button
          variant="secondary"
          className="w-full justify-center"
          onClick={() => {
            router.push("/");
            onNavigate?.();
          }}
        >
          <PlusIcon className="size-4" />
          New agent
          <kbd className="ml-auto font-sans text-[10px] text-ink-faint">⌘K</kbd>
        </Button>
      </div>

      <div className="px-3 pb-2">
        <div className="flex h-8 items-center gap-2 rounded-md border border-hairline bg-panel px-2.5 focus-within:border-hairline-strong">
          <SearchIcon className="size-3.5 shrink-0 text-ink-faint" />
          <input
            ref={searchRef}
            value={query}
            placeholder="Search agents"
            onChange={(event) => setQuery(event.target.value)}
            className="min-w-0 flex-1 bg-transparent text-[12.5px] text-ink outline-none"
          />
          {query ? (
            <button
              type="button"
              onClick={() => setQuery("")}
              className="text-[11px] text-ink-faint hover:text-ink"
            >
              Clear
            </button>
          ) : null}
        </div>
      </div>

      <nav
        className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2 pb-2"
        aria-label="Agents"
      >
        {agentsLoading && agents.length === 0 ? (
          <div className="flex items-center justify-center gap-2 py-10 text-[12px] text-ink-faint">
            <SpinnerIcon className="size-3.5 animate-spin-slow" />
            Loading agents
          </div>
        ) : null}

        {!agentsLoading && filtered.length === 0 ? (
          <p className="px-2 py-10 text-center text-[12px] leading-relaxed text-ink-faint">
            {query
              ? "No agents match that search."
              : "No agents yet. Start one to see it here."}
          </p>
        ) : null}

        {running.length > 0 ? (
          <>
            <p className="px-2 pt-2 pb-1 text-[10px] font-semibold tracking-[0.08em] text-ink-faint uppercase">
              Active
            </p>
            {running.map((agent) => (
              <AgentRow
                key={agent.agentId}
                agent={agent}
                active={agent.agentId === activeAgentId}
              />
            ))}
          </>
        ) : null}

        {rest.length > 0 ? (
          <>
            {running.length > 0 ? (
              <p className="px-2 pt-3 pb-1 text-[10px] font-semibold tracking-[0.08em] text-ink-faint uppercase">
                Recent
              </p>
            ) : null}
            {rest.map((agent) => (
              <AgentRow
                key={agent.agentId}
                agent={agent}
                active={agent.agentId === activeAgentId}
              />
            ))}
          </>
        ) : null}
      </nav>

      <div className="flex items-center gap-2 border-t border-hairline px-3 py-2">
        <button
          type="button"
          onClick={() => {
            setShowArchived(!showArchived);
            void refreshAgents();
          }}
          className={classNames(
            "inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[11px] transition-colors",
            showArchived
              ? "bg-raised text-ink"
              : "text-ink-faint hover:bg-raised hover:text-ink-muted",
          )}
        >
          <ArchiveIcon className="size-3.5" />
          Archived
        </button>

        {session.fromEnvironment ? (
          <span
            className="ml-auto inline-flex items-center gap-1 text-[11px] text-ink-faint"
            title="Authenticated with the CURSOR_API_KEY environment variable"
          >
            <KeyIcon className="size-3" />
            env key
          </span>
        ) : null}
      </div>
    </div>
  );
}
