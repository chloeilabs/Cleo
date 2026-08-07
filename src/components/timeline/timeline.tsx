"use client";

import { useState } from "react";

import { CodeBlock } from "@/components/content/code-block";
import { Markdown } from "@/components/content/markdown";
import {
  BrainIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  TerminalIcon,
} from "@/components/icons";
import { ToolCard } from "@/components/timeline/tool-card";
import { classNames, duration } from "@/lib/format";
import type { TimelineItem } from "@/lib/types";

function ThinkingBlock({
  text,
  durationMs,
}: {
  text: string;
  durationMs?: number;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="overflow-hidden rounded-lg border border-hairline bg-panel">
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        className="flex w-full items-center gap-2.5 px-3 py-2 text-left transition-colors hover:bg-raised"
      >
        <BrainIcon className="size-4 shrink-0 text-thinking" />
        <span className="flex-1 text-[13px] font-medium text-ink-muted">
          Thought
          {durationMs ? (
            <span className="ml-1.5 font-mono text-[11px] text-ink-faint">
              for {duration(durationMs)}
            </span>
          ) : null}
        </span>
        <span className="text-ink-faint">
          {expanded ? (
            <ChevronDownIcon className="size-3.5" />
          ) : (
            <ChevronRightIcon className="size-3.5" />
          )}
        </span>
      </button>

      {expanded ? (
        <div className="border-t border-hairline px-3 py-2.5">
          <p className="text-[13px] leading-[1.7] whitespace-pre-wrap text-ink-faint italic">
            {text}
          </p>
        </div>
      ) : null}
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-xl rounded-br-sm border border-hairline bg-raised px-3.5 py-2.5">
        <p className="text-[14px] leading-[1.65] whitespace-pre-wrap text-ink">
          {text}
        </p>
      </div>
    </div>
  );
}

function ShellTurn({
  item,
}: {
  item: Extract<TimelineItem, { kind: "shell" }>;
}) {
  const output = [item.stdout?.trimEnd(), item.stderr?.trimEnd()]
    .filter(Boolean)
    .join("\n");

  return (
    <div className="overflow-hidden rounded-lg border border-hairline bg-panel">
      <div className="flex items-center gap-2.5 px-3 py-2">
        <TerminalIcon className="size-4 shrink-0 text-ink-faint" />
        <span className="min-w-0 flex-1 truncate font-mono text-[12.5px] text-ink">
          {item.command}
        </span>
        {item.exitCode !== undefined ? (
          <span
            className={classNames(
              "shrink-0 font-mono text-[11px]",
              item.exitCode === 0 ? "text-ink-faint" : "text-negative",
            )}
          >
            exit {item.exitCode}
          </span>
        ) : null}
      </div>

      {output ? (
        <div className="border-t border-hairline p-2">
          <CodeBlock code={output} language="bash" maxLines={16} />
        </div>
      ) : null}
    </div>
  );
}

const STATUS_COPY: Record<string, string> = {
  CREATING: "Provisioning a machine and cloning the repository",
  RUNNING: "Agent is working",
  FINISHED: "Run finished",
  ERROR: "Run failed",
  CANCELLED: "Run cancelled",
  EXPIRED: "Run expired",
};

function StatusRow({
  item,
}: {
  item: Extract<TimelineItem, { kind: "status" }>;
}) {
  const terminal = ["FINISHED", "ERROR", "CANCELLED", "EXPIRED"].includes(
    item.status,
  );

  return (
    <div className="flex items-center gap-2 py-0.5 pl-1">
      <span
        className={classNames(
          "size-1.5 shrink-0 rounded-full",
          item.status === "ERROR" ? "bg-negative" : "",
          item.status === "FINISHED" ? "bg-positive" : "",
          !terminal ? "animate-pulse-dot bg-accent" : "",
          item.status === "CANCELLED" || item.status === "EXPIRED"
            ? "bg-ink-faint"
            : "",
        )}
      />
      <span className="text-[12px] text-ink-faint">
        {item.message ?? STATUS_COPY[item.status] ?? item.status}
      </span>
    </div>
  );
}

export function TimelineRow({ item }: { item: TimelineItem }) {
  switch (item.kind) {
    case "user":
      return <UserBubble text={item.text} />;

    case "assistant":
      return <Markdown>{item.text}</Markdown>;

    case "thinking":
      return <ThinkingBlock text={item.text} durationMs={item.durationMs} />;

    case "tool":
      return <ToolCard tool={item.tool} />;

    case "shell":
      return <ShellTurn item={item} />;

    case "status":
      return <StatusRow item={item} />;

    case "error":
      return (
        <div className="rounded-lg border border-negative/30 bg-negative/[0.06] px-3 py-2.5">
          <p className="text-[13px] text-negative">{item.message}</p>
          {item.code ? (
            <p className="mt-0.5 font-mono text-[11px] text-negative/70">
              {item.code}
            </p>
          ) : null}
        </div>
      );

    // Token usage is surfaced in the run footer and the details panel rather
    // than as a timeline row.
    case "usage":
      return null;
  }
}
