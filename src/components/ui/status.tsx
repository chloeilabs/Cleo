"use client";

import { classNames } from "@/lib/format";
import type { RunStatus } from "@/lib/types";

export type AgentStatus =
  | "running"
  | "finished"
  | "error"
  | "cancelled"
  | "archived"
  | "idle";

const DOT_COLORS: Record<AgentStatus, string> = {
  running: "bg-accent",
  finished: "bg-positive",
  error: "bg-negative",
  cancelled: "bg-caution",
  archived: "bg-ink-faint",
  idle: "bg-ink-faint",
};

const LABELS: Record<AgentStatus, string> = {
  running: "Working",
  finished: "Done",
  error: "Failed",
  cancelled: "Cancelled",
  archived: "Archived",
  idle: "Idle",
};

export function StatusDot({
  status,
  className,
}: {
  status: AgentStatus;
  className?: string;
}) {
  return (
    <span
      className={classNames(
        "relative inline-flex size-2 shrink-0",
        className,
      )}
      title={LABELS[status]}
    >
      {status === "running" ? (
        <span className="absolute inline-flex size-full animate-ping rounded-full bg-accent opacity-60" />
      ) : null}
      <span
        className={classNames(
          "relative inline-flex size-2 rounded-full",
          DOT_COLORS[status],
        )}
      />
    </span>
  );
}

const BADGE_STYLES: Record<AgentStatus, string> = {
  running: "border-accent/30 bg-accent/10 text-accent",
  finished: "border-positive/25 bg-positive/10 text-positive",
  error: "border-negative/30 bg-negative/10 text-negative",
  cancelled: "border-caution/30 bg-caution/10 text-caution",
  archived: "border-hairline bg-raised text-ink-faint",
  idle: "border-hairline bg-raised text-ink-muted",
};

export function StatusBadge({
  status,
  label,
  className,
}: {
  status: AgentStatus;
  label?: string;
  className?: string;
}) {
  return (
    <span
      className={classNames(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium",
        BADGE_STYLES[status],
        className,
      )}
    >
      <StatusDot status={status} />
      {label ?? LABELS[status]}
    </span>
  );
}

export function agentStatus(input: {
  status?: "running" | "finished" | "error";
  archived?: boolean;
}): AgentStatus {
  if (input.archived) return "archived";
  return input.status ?? "idle";
}

export function runStatusToAgentStatus(status: RunStatus): AgentStatus {
  switch (status) {
    case "running":
      return "running";
    case "finished":
      return "finished";
    case "error":
      return "error";
    case "cancelled":
      return "cancelled";
  }
}
