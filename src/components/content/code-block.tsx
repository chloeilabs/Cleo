"use client";

import { useMemo, useState } from "react";

import { CheckIcon, CopyIcon } from "@/components/icons";
import { classNames } from "@/lib/format";
import { TOKEN_CLASS, tokenize } from "@/lib/highlight";

export function CopyButton({
  value,
  className,
  label = "Copy",
}: {
  value: string;
  className?: string;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      type="button"
      aria-label={copied ? "Copied" : label}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1400);
        } catch {
          // Clipboard access can be denied; the button simply does nothing.
        }
      }}
      className={classNames(
        "inline-flex size-6 items-center justify-center rounded text-ink-faint transition-colors hover:bg-raised hover:text-ink",
        className,
      )}
    >
      {copied ? (
        <CheckIcon className="size-3.5 text-positive" />
      ) : (
        <CopyIcon className="size-3.5" />
      )}
    </button>
  );
}

interface CodeBlockProps {
  code: string;
  language?: string;
  /** Show gutter line numbers. */
  numbered?: boolean;
  /** Collapse to this many lines with a "show all" control. */
  maxLines?: number;
  className?: string;
  /** Rendered on the header row, left of the copy button. */
  caption?: string;
}

export function CodeBlock({
  code,
  language,
  numbered = false,
  maxLines,
  className,
  caption,
}: CodeBlockProps) {
  const [expanded, setExpanded] = useState(false);

  const lines = useMemo(() => code.replace(/\n$/, "").split("\n"), [code]);
  const clipped = maxLines !== undefined && !expanded && lines.length > maxLines;
  const visible = clipped ? lines.slice(0, maxLines) : lines;

  const highlighted = useMemo(
    () => visible.map((line) => tokenize(line, language ?? "")),
    [visible, language],
  );

  const gutterWidth = String(lines.length).length;

  return (
    <div
      className={classNames(
        "group/code overflow-hidden rounded-lg border border-hairline bg-canvas",
        className,
      )}
    >
      {caption || language ? (
        <div className="flex items-center justify-between border-b border-hairline bg-panel px-3 py-1.5">
          <span className="truncate font-mono text-[11px] text-ink-faint">
            {caption ?? language}
          </span>
          <CopyButton value={code} />
        </div>
      ) : (
        <div className="pointer-events-none absolute" />
      )}

      <div className="relative">
        {!caption && !language ? (
          <CopyButton
            value={code}
            className="absolute top-1.5 right-1.5 z-10 opacity-0 transition-opacity group-hover/code:opacity-100"
          />
        ) : null}

        <pre className="overflow-x-auto px-3 py-2.5 font-mono text-[12.5px] leading-[1.65]">
          <code>
            {highlighted.map((tokens, lineIndex) => (
              <div key={lineIndex} className="flex min-w-fit">
                {numbered ? (
                  <span
                    aria-hidden
                    className="mr-3 shrink-0 text-right text-ink-faint/60 select-none"
                    style={{ width: `${gutterWidth}ch` }}
                  >
                    {lineIndex + 1}
                  </span>
                ) : null}
                <span className="whitespace-pre">
                  {tokens.length === 0 ? " " : null}
                  {tokens.map((token, tokenIndex) => (
                    <span key={tokenIndex} className={TOKEN_CLASS[token.kind]}>
                      {token.text}
                    </span>
                  ))}
                </span>
              </div>
            ))}
          </code>
        </pre>

        {clipped ? (
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="w-full border-t border-hairline bg-panel py-1.5 text-[11px] text-ink-muted transition-colors hover:text-ink"
          >
            Show {lines.length - maxLines} more lines
          </button>
        ) : null}
      </div>
    </div>
  );
}
