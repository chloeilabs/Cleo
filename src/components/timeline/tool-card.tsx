"use client";

import { type ComponentType, type SVGProps, useState } from "react";

import { CodeBlock } from "@/components/content/code-block";
import { DiffView } from "@/components/content/diff-view";
import { Markdown } from "@/components/content/markdown";
import {
  BranchIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CloseIcon,
  FileIcon,
  GlobeIcon,
  ImageIcon,
  ListIcon,
  PencilIcon,
  SearchCodeIcon,
  SearchIcon,
  SparkIcon,
  SpinnerIcon,
  TerminalIcon,
  TrashIcon,
} from "@/components/icons";
import { classNames } from "@/lib/format";
import type { ToolView } from "@/lib/types";

type IconType = ComponentType<SVGProps<SVGSVGElement>>;

interface ToolPresentation {
  label: string;
  Icon: IconType;
  /** Render the title in monospace (paths, commands, patterns). */
  mono?: boolean;
}

const PRESENTATION: Record<string, ToolPresentation> = {
  shell: { label: "Ran", Icon: TerminalIcon, mono: true },
  read: { label: "Read", Icon: FileIcon, mono: true },
  edit: { label: "Edited", Icon: PencilIcon, mono: true },
  write: { label: "Wrote", Icon: PencilIcon, mono: true },
  delete: { label: "Deleted", Icon: TrashIcon, mono: true },
  ls: { label: "Listed", Icon: ListIcon, mono: true },
  glob: { label: "Found files", Icon: SearchIcon, mono: true },
  grep: { label: "Searched", Icon: SearchIcon, mono: true },
  semSearch: { label: "Searched the codebase", Icon: SearchCodeIcon },
  task: { label: "Delegated to a subagent", Icon: BranchIcon },
  updateTodos: { label: "Plan", Icon: ListIcon },
  createPlan: { label: "Plan", Icon: ListIcon },
  mcp: { label: "MCP", Icon: SparkIcon },
  readLints: { label: "Checked lints", Icon: CheckIcon },
  generateImage: { label: "Generated an image", Icon: ImageIcon },
  recordScreen: { label: "Recorded the screen", Icon: ImageIcon },
  webSearch: { label: "Searched the web", Icon: GlobeIcon },
  webFetch: { label: "Fetched", Icon: GlobeIcon, mono: true },
};

function presentationFor(name: string): ToolPresentation {
  return PRESENTATION[name] ?? { label: name, Icon: SparkIcon };
}

function bodyLanguage(tool: ToolView): string {
  if (tool.name === "shell") return "bash";
  const extension = tool.title.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    ts: "typescript",
    tsx: "typescript",
    js: "javascript",
    jsx: "javascript",
    py: "python",
    rb: "ruby",
    go: "go",
    rs: "rust",
    sh: "bash",
    json: "json",
    yml: "yaml",
    yaml: "yaml",
    css: "css",
    html: "html",
    md: "markdown",
  };
  return map[extension] ?? "";
}

const TODO_STYLES = {
  completed: "text-ink-faint line-through decoration-ink-faint/60",
  inProgress: "text-ink",
  pending: "text-ink-muted",
  cancelled: "text-ink-faint line-through decoration-ink-faint/60",
} as const;

function TodoList({ tool }: { tool: ToolView }) {
  if (!tool.todos?.length) return null;

  return (
    <ul className="space-y-1.5">
      {tool.todos.map((todo, index) => (
        <li key={index} className="flex items-start gap-2 text-[13px]">
          <span
            className={classNames(
              "mt-[3px] flex size-3.5 shrink-0 items-center justify-center rounded-[4px] border",
              todo.status === "completed"
                ? "border-positive/50 bg-positive/20 text-positive"
                : todo.status === "inProgress"
                  ? "border-accent bg-accent/20 text-accent"
                  : todo.status === "cancelled"
                    ? "border-hairline-strong text-ink-faint"
                    : "border-hairline-strong",
            )}
          >
            {todo.status === "completed" ? (
              <CheckIcon className="size-2.5" strokeWidth={3} />
            ) : todo.status === "inProgress" ? (
              <span className="size-1.5 rounded-full bg-accent" />
            ) : todo.status === "cancelled" ? (
              <CloseIcon className="size-2.5" strokeWidth={3} />
            ) : null}
          </span>
          <span className={TODO_STYLES[todo.status]}>{todo.content}</span>
        </li>
      ))}
    </ul>
  );
}

export function ToolCard({ tool }: { tool: ToolView }) {
  const { label, Icon, mono } = presentationFor(tool.name);

  const isPlan = tool.name === "updateTodos" || tool.name === "createPlan";
  const hasDetail = Boolean(
    tool.body || tool.diff || tool.markdown || tool.rawArgs || tool.rawResult,
  );

  // Short diffs read better inline, which is also how Cursor shows them.
  const diffLines = tool.diff ? tool.diff.split("\n").length : 0;
  const [expanded, setExpanded] = useState(
    isPlan || (diffLines > 0 && diffLines <= 22),
  );
  const [showRaw, setShowRaw] = useState(false);

  const interactive = hasDetail && !isPlan;

  return (
    <div
      className={classNames(
        "overflow-hidden rounded-lg border transition-colors",
        tool.status === "error"
          ? "border-negative/30 bg-negative/[0.04]"
          : "border-hairline bg-panel",
      )}
    >
      <button
        type="button"
        disabled={!interactive}
        onClick={() => setExpanded((value) => !value)}
        className={classNames(
          "flex w-full items-center gap-2.5 px-3 py-2 text-left",
          interactive && "transition-colors hover:bg-raised",
          !interactive && "cursor-default",
        )}
      >
        <span
          className={classNames(
            "shrink-0",
            tool.status === "error" ? "text-negative" : "text-ink-faint",
          )}
        >
          {tool.status === "running" ? (
            <SpinnerIcon className="size-4 animate-spin-slow text-accent" />
          ) : (
            <Icon className="size-4" />
          )}
        </span>

        <span className="shrink-0 text-[13px] font-medium text-ink-muted">
          {label}
        </span>

        {isPlan ? null : (
          <span
            className={classNames(
              "min-w-0 flex-1 truncate text-[13px]",
              mono ? "font-mono text-[12.5px]" : "",
              tool.status === "error" ? "text-negative" : "text-ink",
            )}
            title={tool.title}
          >
            {tool.title}
          </span>
        )}

        {tool.subtitle ? (
          <span
            className={classNames(
              "shrink-0 font-mono text-[11px]",
              isPlan ? "flex-1 text-left" : "",
              tool.status === "error" ? "text-negative/80" : "text-ink-faint",
            )}
          >
            {tool.subtitle}
          </span>
        ) : null}

        {interactive ? (
          <span className="shrink-0 text-ink-faint">
            {expanded ? (
              <ChevronDownIcon className="size-3.5" />
            ) : (
              <ChevronRightIcon className="size-3.5" />
            )}
          </span>
        ) : null}
      </button>

      {expanded ? (
        <div className="space-y-2 border-t border-hairline px-3 py-2.5">
          {tool.todos ? <TodoList tool={tool} /> : null}

          {tool.markdown ? (
            <Markdown className="text-[13px]">{tool.markdown}</Markdown>
          ) : null}

          {tool.diff ? (
            <DiffView diff={tool.diff} path={tool.title} maxRows={26} />
          ) : null}

          {tool.errorText && !tool.body ? (
            <p className="font-mono text-[12px] text-negative">
              {tool.errorText}
            </p>
          ) : null}

          {tool.body ? (
            <CodeBlock
              code={tool.body}
              language={bodyLanguage(tool)}
              maxLines={20}
            />
          ) : null}

          {tool.truncated ? (
            <p className="text-[11px] text-ink-faint">
              Cursor truncated this payload because it was too large.
            </p>
          ) : null}

          {tool.rawArgs || tool.rawResult ? (
            <div>
              <button
                type="button"
                onClick={() => setShowRaw((value) => !value)}
                className="text-[11px] text-ink-faint transition-colors hover:text-ink-muted"
              >
                {showRaw ? "Hide" : "Show"} raw payload
              </button>

              {showRaw ? (
                <div className="mt-2 space-y-2">
                  {tool.rawArgs ? (
                    <CodeBlock
                      code={tool.rawArgs}
                      language="json"
                      caption="args"
                      maxLines={24}
                    />
                  ) : null}
                  {tool.rawResult ? (
                    <CodeBlock
                      code={tool.rawResult}
                      language="json"
                      caption="result"
                      maxLines={24}
                    />
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
