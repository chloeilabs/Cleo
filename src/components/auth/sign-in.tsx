"use client";

import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";

import { ExternalIcon, KeyIcon, LogoIcon } from "@/components/icons";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

const HIGHLIGHTS = [
  "Dispatch agents into isolated cloud machines with your repo cloned in",
  "Watch reasoning, shell output, and diffs stream in live",
  "Follow up with full conversation context, then ship a pull request",
];

export function SignIn() {
  const router = useRouter();
  const [apiKey, setApiKey] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!apiKey.trim() || submitting) return;

    setSubmitting(true);
    setError(null);

    try {
      await api.signIn(apiKey.trim());
      router.refresh();
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "That key could not be verified.",
      );
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-full items-center justify-center px-5 py-12">
      <div className="w-full max-w-sm">
        <div className="mb-7 text-center">
          <LogoIcon className="mx-auto mb-4 size-12 text-accent" />
          <h1 className="text-[22px] font-semibold tracking-tight text-ink">
            Cleo
          </h1>
          <p className="mt-1.5 text-[13.5px] leading-relaxed text-ink-muted">
            A cloud coding agent workspace, built on the Cursor Agent SDK.
          </p>
        </div>

        <form
          onSubmit={submit}
          className="rounded-xl border border-hairline bg-panel p-4"
        >
          <label
            htmlFor="api-key"
            className="mb-1.5 block text-[12px] font-medium text-ink-muted"
          >
            Cursor API key
          </label>

          <div className="flex h-9 items-center gap-2 rounded-md border border-hairline bg-canvas px-2.5 focus-within:border-accent/60">
            <KeyIcon className="size-3.5 shrink-0 text-ink-faint" />
            <input
              id="api-key"
              type="password"
              value={apiKey}
              autoComplete="off"
              spellCheck={false}
              placeholder="key_..."
              onChange={(event) => setApiKey(event.target.value)}
              className="min-w-0 flex-1 bg-transparent font-mono text-[13px] text-ink outline-none"
            />
          </div>

          {error ? (
            <p className="mt-2 text-[12px] leading-relaxed text-negative">
              {error}
            </p>
          ) : null}

          <Button
            type="submit"
            variant="primary"
            loading={submitting}
            disabled={!apiKey.trim()}
            className="mt-3 w-full justify-center"
          >
            Continue
          </Button>

          <p className="mt-3 text-[11.5px] leading-relaxed text-ink-faint">
            The key is encrypted into an http-only cookie and only ever leaves
            the server to talk to Cursor. Set{" "}
            <code className="font-mono text-ink-muted">CURSOR_API_KEY</code> to
            skip this screen.
          </p>
        </form>

        <a
          href="https://cursor.com/dashboard/api"
          target="_blank"
          rel="noreferrer"
          className="mt-3 inline-flex w-full items-center justify-center gap-1.5 text-[12.5px] text-accent hover:underline"
        >
          Create an API key
          <ExternalIcon className="size-3" />
        </a>

        <ul className="mt-7 space-y-2">
          {HIGHLIGHTS.map((highlight) => (
            <li
              key={highlight}
              className="flex items-start gap-2 text-[12.5px] leading-relaxed text-ink-faint"
            >
              <span className="mt-[7px] size-1 shrink-0 rounded-full bg-accent" />
              {highlight}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
