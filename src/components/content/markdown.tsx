"use client";

import { memo, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { CodeBlock } from "@/components/content/code-block";
import { classNames } from "@/lib/format";

function toText(children: ReactNode): string {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(toText).join("");
  return "";
}

const COMPONENTS: Components = {
  p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,

  h1: ({ children }) => (
    <h1 className="mt-5 mb-2 text-base font-semibold text-ink first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="mt-5 mb-2 text-[15px] font-semibold text-ink first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-4 mb-1.5 text-sm font-semibold text-ink first:mt-0">
      {children}
    </h3>
  ),
  h4: ({ children }) => (
    <h4 className="mt-4 mb-1.5 text-sm font-medium text-ink first:mt-0">
      {children}
    </h4>
  ),

  ul: ({ children }) => (
    <ul className="my-2 ml-1 list-outside list-disc space-y-1 pl-4 marker:text-ink-faint">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="my-2 ml-1 list-outside list-decimal space-y-1 pl-4 marker:text-ink-faint">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="pl-0.5">{children}</li>,

  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent"
    >
      {children}
    </a>
  ),

  strong: ({ children }) => (
    <strong className="font-semibold text-ink">{children}</strong>
  ),
  em: ({ children }) => <em className="italic">{children}</em>,

  hr: () => <hr className="my-4 border-hairline" />,

  blockquote: ({ children }) => (
    <blockquote className="my-3 border-l-2 border-hairline-strong pl-3 text-ink-muted">
      {children}
    </blockquote>
  ),

  table: ({ children }) => (
    <div className="my-3 overflow-x-auto rounded-lg border border-hairline">
      <table className="w-full border-collapse text-[13px]">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-panel">{children}</thead>,
  th: ({ children }) => (
    <th className="border-b border-hairline px-3 py-2 text-left font-medium text-ink">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border-b border-hairline/60 px-3 py-2 align-top text-ink-muted">
      {children}
    </td>
  ),

  code: ({ className, children, ...props }) => {
    const language = /language-(\w+)/.exec(className ?? "")?.[1];
    const text = toText(children);

    // `react-markdown` routes both inline spans and fenced blocks here; a
    // fence is the case that carries a language or spans multiple lines.
    if (!language && !text.includes("\n")) {
      return (
        <code
          className="rounded border border-hairline bg-raised px-1 py-0.5 font-mono text-[12px] text-ink"
          {...props}
        >
          {children}
        </code>
      );
    }

    return (
      <CodeBlock
        code={text.replace(/\n$/, "")}
        language={language}
        maxLines={40}
        className="my-3"
      />
    );
  },

  pre: ({ children }) => <>{children}</>,
};

/** Assistant and plan prose. GFM is on for tables and task lists. */
export const Markdown = memo(function Markdown({
  children,
  className,
}: {
  children: string;
  className?: string;
}) {
  return (
    <div
      className={classNames(
        "text-[14px] leading-[1.7] break-words text-ink-muted [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        className,
      )}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {children}
      </ReactMarkdown>
    </div>
  );
});
