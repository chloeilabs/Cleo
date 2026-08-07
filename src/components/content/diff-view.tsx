"use client";

import { useMemo, useState } from "react";

import { CopyButton } from "@/components/content/code-block";
import { classNames } from "@/lib/format";
import { TOKEN_CLASS, tokenize } from "@/lib/highlight";

type DiffLineKind = "added" | "removed" | "context" | "hunk" | "meta";

interface DiffLine {
  kind: DiffLineKind;
  text: string;
  oldLine?: number;
  newLine?: number;
}

const HUNK = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/;

/**
 * Parse unified diff text into rows carrying both sides' line numbers, so the
 * gutter can show the real positions rather than a running index.
 */
function parseDiff(diff: string): DiffLine[] {
  const rows: DiffLine[] = [];
  let oldLine = 0;
  let newLine = 0;

  for (const raw of diff.replace(/\n$/, "").split("\n")) {
    const hunk = HUNK.exec(raw);
    if (hunk) {
      oldLine = Number(hunk[1]);
      newLine = Number(hunk[2]);
      rows.push({ kind: "hunk", text: raw });
      continue;
    }

    if (
      raw.startsWith("diff --git") ||
      raw.startsWith("index ") ||
      raw.startsWith("--- ") ||
      raw.startsWith("+++ ") ||
      raw.startsWith("new file mode") ||
      raw.startsWith("deleted file mode") ||
      raw.startsWith("similarity index") ||
      raw.startsWith("rename ")
    ) {
      rows.push({ kind: "meta", text: raw });
      continue;
    }

    if (raw.startsWith("+")) {
      rows.push({ kind: "added", text: raw.slice(1), newLine: newLine++ });
      continue;
    }

    if (raw.startsWith("-")) {
      rows.push({ kind: "removed", text: raw.slice(1), oldLine: oldLine++ });
      continue;
    }

    if (raw.startsWith("\\")) {
      rows.push({ kind: "meta", text: raw });
      continue;
    }

    rows.push({
      kind: "context",
      text: raw.startsWith(" ") ? raw.slice(1) : raw,
      oldLine: oldLine++,
      newLine: newLine++,
    });
  }

  return rows;
}

const ROW_STYLES: Record<DiffLineKind, string> = {
  added: "bg-positive/10",
  removed: "bg-negative/10",
  context: "",
  hunk: "bg-accent-soft/60",
  meta: "",
};

const TEXT_STYLES: Record<DiffLineKind, string> = {
  added: "text-ink",
  removed: "text-ink-muted",
  context: "text-ink-muted",
  hunk: "text-accent",
  meta: "text-ink-faint",
};

const MARKERS: Record<DiffLineKind, string> = {
  added: "+",
  removed: "−",
  context: " ",
  hunk: " ",
  meta: " ",
};

function languageFromPath(path?: string): string {
  const extension = path?.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    ts: "typescript",
    tsx: "typescript",
    js: "javascript",
    jsx: "javascript",
    mjs: "javascript",
    cjs: "javascript",
    py: "python",
    rb: "ruby",
    go: "go",
    rs: "rust",
    sh: "bash",
    bash: "bash",
    zsh: "bash",
    json: "json",
    yml: "yaml",
    yaml: "yaml",
    toml: "toml",
    html: "html",
    css: "css",
    md: "markdown",
  };
  return map[extension] ?? "";
}

export function DiffView({
  diff,
  path,
  maxRows = 24,
}: {
  diff: string;
  path?: string;
  maxRows?: number;
}) {
  const [expanded, setExpanded] = useState(false);

  const rows = useMemo(() => parseDiff(diff), [diff]);
  const language = useMemo(() => languageFromPath(path), [path]);

  const clipped = !expanded && rows.length > maxRows;
  const visible = clipped ? rows.slice(0, maxRows) : rows;

  const gutterWidth = Math.max(
    2,
    String(
      rows.reduce(
        (max, row) => Math.max(max, row.oldLine ?? 0, row.newLine ?? 0),
        0,
      ),
    ).length,
  );

  return (
    <div className="group/diff relative overflow-hidden rounded-lg border border-hairline bg-canvas">
      <CopyButton
        value={diff}
        className="absolute top-1.5 right-1.5 z-10 opacity-0 transition-opacity group-hover/diff:opacity-100"
        label="Copy diff"
      />

      <div className="overflow-x-auto font-mono text-[12.5px] leading-[1.6]">
        {visible.map((row, index) => (
          <div
            key={index}
            className={classNames("flex min-w-fit", ROW_STYLES[row.kind])}
          >
            <span
              aria-hidden
              className="shrink-0 border-r border-hairline/60 px-2 text-right text-ink-faint/50 select-none"
              style={{ width: `calc(${gutterWidth}ch + 1rem)` }}
            >
              {row.oldLine ?? ""}
            </span>
            <span
              aria-hidden
              className="shrink-0 border-r border-hairline/60 px-2 text-right text-ink-faint/50 select-none"
              style={{ width: `calc(${gutterWidth}ch + 1rem)` }}
            >
              {row.newLine ?? ""}
            </span>
            <span
              aria-hidden
              className={classNames(
                "w-5 shrink-0 pl-2 select-none",
                row.kind === "added" ? "text-positive" : "",
                row.kind === "removed" ? "text-negative" : "",
                row.kind === "context" || row.kind === "meta"
                  ? "text-ink-faint/40"
                  : "",
              )}
            >
              {MARKERS[row.kind]}
            </span>
            <span
              className={classNames(
                "pr-3 whitespace-pre",
                TEXT_STYLES[row.kind],
              )}
            >
              {row.kind === "hunk" || row.kind === "meta"
                ? row.text
                : tokenize(row.text, language).map((token, tokenIndex) => (
                    <span
                      key={tokenIndex}
                      className={
                        row.kind === "removed"
                          ? "opacity-70"
                          : TOKEN_CLASS[token.kind]
                      }
                    >
                      {token.text}
                    </span>
                  ))}
              {row.text === "" ? " " : null}
            </span>
          </div>
        ))}
      </div>

      {clipped ? (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="w-full border-t border-hairline bg-panel py-1.5 text-[11px] text-ink-muted transition-colors hover:text-ink"
        >
          Show {rows.length - maxRows} more lines
        </button>
      ) : null}
    </div>
  );
}
