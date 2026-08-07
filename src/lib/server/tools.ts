import "server-only";

import type { TodoEntry, ToolStatus, ToolView } from "@/lib/types";

/**
 * Tool `args` and `result` payloads are explicitly unstable in the Cursor SDK,
 * and the live stream types them as `unknown`. Everything here reads them
 * defensively and degrades to pretty-printed JSON rather than throwing, so a
 * schema change downgrades the rendering instead of breaking the timeline.
 */

const BODY_LIMIT = 12_000;
const DIFF_LIMIT = 20_000;
const RAW_LIMIT = 20_000;

type Rec = Record<string, unknown>;

function asRecord(value: unknown): Rec {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Rec)
    : {};
}

function str(record: Rec, key: string): string | undefined {
  const value = record[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function num(record: Rec, key: string): number | undefined {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function clip(value: string | undefined, limit: number): string | undefined {
  if (!value) return undefined;
  if (value.length <= limit) return value;
  return `${value.slice(0, limit)}\n… ${value.length - limit} more characters`;
}

function pretty(value: unknown, limit = RAW_LIMIT): string | undefined {
  if (value === undefined || value === null) return undefined;
  try {
    const text =
      typeof value === "string" ? value : JSON.stringify(value, null, 2);
    return clip(text, limit);
  } catch {
    return undefined;
  }
}

interface UnwrappedResult {
  ok: boolean;
  value: Rec;
  errorText?: string;
}

/**
 * Tool results arrive as `{ status: "success", value }` / `{ status: "error",
 * error }` from the conversation API, but the live stream sometimes carries the
 * bare value. Accept both.
 */
function unwrapResult(result: unknown): UnwrappedResult | undefined {
  if (result === undefined || result === null) return undefined;

  const record = asRecord(result);
  const status = str(record, "status");

  if (status === "error") {
    const error = record.error;
    return {
      ok: false,
      value: asRecord(error),
      errorText:
        typeof error === "string"
          ? error
          : (str(asRecord(error), "message") ??
            str(asRecord(error), "reason") ??
            pretty(error, 2_000)),
    };
  }

  if (status === "success" || "value" in record) {
    return { ok: true, value: asRecord(record.value) };
  }

  return { ok: true, value: record };
}

function joinStreams(stdout?: string, stderr?: string): string | undefined {
  const parts = [stdout?.trimEnd(), stderr?.trimEnd()].filter(
    (part): part is string => Boolean(part),
  );
  return parts.length > 0 ? parts.join("\n") : undefined;
}

function fileName(path: string | undefined): string | undefined {
  if (!path) return undefined;
  return path.split("/").filter(Boolean).at(-1) ?? path;
}

function toTodos(value: unknown): TodoEntry[] | undefined {
  if (!Array.isArray(value)) return undefined;

  const todos = value.flatMap((entry) => {
    const record = asRecord(entry);
    const content = str(record, "content");
    if (!content) return [];
    const status = str(record, "status");
    return [
      {
        content,
        status:
          status === "inProgress" ||
          status === "completed" ||
          status === "cancelled"
            ? status
            : ("pending" as const),
      } satisfies TodoEntry,
    ];
  });

  return todos.length > 0 ? todos : undefined;
}

function formatGrepMatches(value: Rec): string | undefined {
  const groups: unknown[] = [];
  const workspaceResults = asRecord(value.workspaceResults);
  for (const entry of Object.values(workspaceResults)) groups.push(entry);
  if (value.activeEditorResult) groups.push(value.activeEditorResult);

  const lines: string[] = [];
  for (const group of groups) {
    const record = asRecord(group);
    const output = asRecord(record.output);

    if (Array.isArray(output.matches)) {
      for (const match of output.matches) {
        const matchRecord = asRecord(match);
        const file = str(matchRecord, "file") ?? "";
        const lineNumber = num(matchRecord, "lineNumber");
        const text = str(matchRecord, "line")?.trim() ?? "";
        lines.push(
          `${file}${lineNumber ? `:${lineNumber}` : ""}${text ? `: ${text}` : ""}`,
        );
      }
    }

    if (Array.isArray(output.files)) {
      for (const file of output.files) {
        if (typeof file === "string") lines.push(file);
      }
    }

    if (Array.isArray(output.counts)) {
      for (const count of output.counts) {
        const countRecord = asRecord(count);
        lines.push(
          `${str(countRecord, "file") ?? ""}: ${num(countRecord, "count") ?? 0}`,
        );
      }
    }
  }

  return lines.length > 0 ? lines.join("\n") : undefined;
}

function formatLsTree(node: unknown, depth = 0, lines: string[] = []): string[] {
  const record = asRecord(node);
  const name = str(record, "name") ?? str(record, "path");
  if (name) lines.push(`${"  ".repeat(depth)}${name}`);

  const children = record.children ?? record.entries ?? record.nodes;
  if (Array.isArray(children)) {
    for (const child of children.slice(0, 200)) {
      formatLsTree(child, depth + 1, lines);
    }
  }

  return lines;
}

function formatMcpContent(value: Rec): string | undefined {
  if (!Array.isArray(value.content)) return undefined;

  const texts = value.content.flatMap((entry) => {
    const text = str(asRecord(asRecord(entry).text), "text");
    return text ? [text] : [];
  });

  return texts.length > 0 ? texts.join("\n") : undefined;
}

/** Build the presentational projection for one tool call. */
export function buildToolView(
  name: string,
  status: ToolStatus,
  args: unknown,
  result: unknown,
  truncated?: boolean,
): ToolView {
  const a = asRecord(args);
  const unwrapped = unwrapResult(result);
  const value = unwrapped?.value ?? {};
  const failed = unwrapped?.ok === false;

  const view: ToolView = {
    name,
    status: failed ? "error" : status,
    title: name,
    rawArgs: pretty(args),
    rawResult: pretty(result),
    errorText: unwrapped?.errorText,
    truncated,
  };

  switch (name) {
    case "shell": {
      const exitCode = num(value, "exitCode");
      const executionTime = num(value, "executionTime");
      view.title = str(a, "command") ?? "shell";
      view.subtitle = [
        str(a, "workingDirectory"),
        exitCode === undefined ? undefined : `exit ${exitCode}`,
        executionTime === undefined
          ? undefined
          : `${Math.round(executionTime)}ms`,
      ]
        .filter(Boolean)
        .join(" · ");
      view.body = clip(
        joinStreams(str(value, "stdout"), str(value, "stderr")),
        BODY_LIMIT,
      );
      if (exitCode !== undefined && exitCode !== 0) view.status = "error";
      break;
    }

    case "read": {
      const path = str(a, "path");
      const totalLines = num(value, "totalLines");
      view.title = path ?? "read";
      view.subtitle = totalLines === undefined ? undefined : `${totalLines} lines`;
      view.body = clip(str(value, "content"), BODY_LIMIT);
      break;
    }

    case "edit": {
      const added = num(value, "linesAdded");
      const removed = num(value, "linesRemoved");
      view.title = str(a, "path") ?? "edit";
      view.subtitle =
        added === undefined && removed === undefined
          ? undefined
          : `+${added ?? 0} −${removed ?? 0}`;
      view.diff = clip(str(value, "diffString"), DIFF_LIMIT);
      break;
    }

    case "write": {
      const path = str(a, "path") ?? str(value, "path");
      const linesCreated = num(value, "linesCreated");
      view.title = path ?? "write";
      view.subtitle =
        linesCreated === undefined ? undefined : `${linesCreated} lines`;
      view.body = clip(
        str(a, "fileText") ?? str(value, "fileContentAfterWrite"),
        BODY_LIMIT,
      );
      break;
    }

    case "delete": {
      view.title = str(a, "path") ?? "delete";
      break;
    }

    case "ls": {
      view.title = str(a, "path") ?? "ls";
      const tree = value.directoryTreeRoot;
      view.body = tree ? clip(formatLsTree(tree).join("\n"), BODY_LIMIT) : undefined;
      break;
    }

    case "glob": {
      const totalFiles = num(value, "totalFiles");
      view.title = str(a, "globPattern") ?? "glob";
      view.subtitle = [
        str(a, "targetDirectory"),
        totalFiles === undefined ? undefined : `${totalFiles} files`,
      ]
        .filter(Boolean)
        .join(" · ");
      view.body = Array.isArray(value.files)
        ? clip(value.files.filter((f) => typeof f === "string").join("\n"), BODY_LIMIT)
        : undefined;
      break;
    }

    case "grep": {
      view.title = str(a, "pattern") ?? "grep";
      view.subtitle = [str(a, "path"), str(a, "glob"), str(a, "type")]
        .filter(Boolean)
        .join(" · ");
      view.body = clip(formatGrepMatches(value), BODY_LIMIT);
      break;
    }

    case "semSearch": {
      view.title = str(a, "query") ?? "codebase search";
      view.subtitle = Array.isArray(a.targetDirectories)
        ? a.targetDirectories.filter((d) => typeof d === "string").join(", ")
        : undefined;
      view.body = clip(str(value, "results"), BODY_LIMIT);
      break;
    }

    case "task": {
      view.title = str(a, "description") ?? "subagent";
      view.subtitle = [
        str(asRecord(a.subagentType), "name") ?? str(asRecord(a.subagentType), "kind"),
        str(a, "mode"),
        str(a, "model"),
      ]
        .filter(Boolean)
        .join(" · ");
      view.body = clip(
        str(value, "resultSuffix") ?? str(a, "prompt"),
        BODY_LIMIT,
      );
      break;
    }

    case "updateTodos": {
      const todos = toTodos(value.todos) ?? toTodos(a.todos);
      const done = todos?.filter((todo) => todo.status === "completed").length ?? 0;
      view.title = "Updated the plan";
      view.subtitle = todos ? `${done}/${todos.length} done` : undefined;
      view.todos = todos;
      break;
    }

    case "createPlan": {
      view.title = "Wrote a plan";
      view.markdown = clip(str(a, "plan"), DIFF_LIMIT);
      break;
    }

    case "mcp": {
      const provider = str(a, "providerIdentifier");
      const toolName = str(a, "toolName");
      view.title = [provider, toolName].filter(Boolean).join(" · ") || "MCP tool";
      view.body = clip(formatMcpContent(value), BODY_LIMIT);
      if (value.isError === true) view.status = "error";
      break;
    }

    case "readLints": {
      view.title = "Checked lints";
      view.body = clip(pretty(value, BODY_LIMIT), BODY_LIMIT);
      break;
    }

    case "generateImage": {
      view.title = str(a, "prompt") ?? "Generated an image";
      break;
    }

    case "recordScreen": {
      view.title = "Recorded the screen";
      break;
    }

    default: {
      view.title =
        str(a, "command") ??
        str(a, "query") ??
        str(a, "pattern") ??
        fileName(str(a, "path")) ??
        str(a, "description") ??
        name;
      view.body = clip(pretty(value, BODY_LIMIT), BODY_LIMIT);
      break;
    }
  }

  if (failed && !view.body && view.errorText) {
    view.body = clip(view.errorText, BODY_LIMIT);
  }

  return view;
}
