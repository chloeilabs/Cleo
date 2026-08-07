"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { LogoIcon, RepoIcon, SparkIcon } from "@/components/icons";
import { useToast } from "@/components/ui/toast";
import {
  Composer,
  type ComposerSubmission,
} from "@/components/workspace/composer";
import { modelSelection, useWorkspace } from "@/components/workspace/provider";
import { api } from "@/lib/api";
import { firstLine } from "@/lib/format";

const SUGGESTIONS = [
  {
    title: "Fix the failing tests",
    prompt:
      "Run the test suite, find every failing test, and fix the underlying bugs. Explain each root cause in your summary.",
  },
  {
    title: "Review recent changes",
    prompt:
      "Review the changes on this branch against main. Flag correctness bugs, missing error handling, and anything that would break in production.",
  },
  {
    title: "Write the missing tests",
    prompt:
      "Find the modules with the weakest test coverage and add focused tests for the behaviour that actually matters.",
  },
  {
    title: "Upgrade dependencies",
    prompt:
      "Upgrade dependencies to their latest compatible versions, fix any breaking changes, and make sure the build and tests still pass.",
  },
];

export function NewAgentView() {
  const router = useRouter();
  const { reportError } = useToast();
  const {
    preferences,
    selectedModel,
    repositories,
    refreshAgents,
  } = useWorkspace();

  const [submitting, setSubmitting] = useState(false);
  const [seed, setSeed] = useState<string | undefined>();

  const repository = repositories.find(
    (repo) => repo.url === preferences.repoUrl,
  );

  const create = async ({ prompt, images }: ComposerSubmission) => {
    setSubmitting(true);

    try {
      const { agentId } = await api.createAgent({
        prompt,
        repoUrl: preferences.repoUrl,
        startingRef: preferences.startingRef || undefined,
        model: modelSelection(selectedModel),
        mode: preferences.mode,
        autoCreatePR: preferences.repoUrl ? preferences.autoCreatePR : false,
        workOnCurrentBranch: preferences.workOnCurrentBranch,
        name: firstLine(prompt, 60),
        images: images.length > 0 ? images : undefined,
      });

      void refreshAgents();
      router.push(`/agents/${encodeURIComponent(agentId)}`);
    } catch (error) {
      reportError(error, "The agent could not be started.");
      setSubmitting(false);
    }
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center px-5 py-16">
        <div className="mb-8 text-center">
          <LogoIcon className="mx-auto mb-4 size-11 text-accent" />
          <h1 className="text-[26px] leading-tight font-semibold tracking-tight text-ink">
            What should the agent work on?
          </h1>
          <p className="mt-2 text-[14px] text-ink-muted">
            {repository ? (
              <>
                Runs in an isolated cloud machine on{" "}
                <span className="inline-flex items-center gap-1 font-medium text-ink">
                  <RepoIcon className="size-3.5" />
                  {repository.label}
                </span>
              </>
            ) : (
              "Pick a repository below, or start with an empty cloud workspace."
            )}
          </p>
        </div>

        <Composer
          key={seed}
          variant="create"
          autoFocus
          busy={submitting}
          placeholder="Describe a task — the agent clones the repo, works in its own machine, and pushes a branch."
          onSubmit={create}
        />

        <div className="mt-6">
          <p className="mb-2.5 flex items-center gap-1.5 text-[11px] font-semibold tracking-[0.08em] text-ink-faint uppercase">
            <SparkIcon className="size-3" />
            Try one of these
          </p>

          <div className="grid gap-2 sm:grid-cols-2">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion.title}
                type="button"
                disabled={submitting}
                onClick={() => {
                  setSeed(suggestion.title);
                  void create({ prompt: suggestion.prompt, images: [] });
                }}
                className="rounded-lg border border-hairline bg-panel px-3.5 py-3 text-left transition-colors hover:border-hairline-strong hover:bg-raised disabled:cursor-not-allowed disabled:opacity-60"
              >
                <p className="text-[13px] font-medium text-ink">
                  {suggestion.title}
                </p>
                <p className="mt-1 line-clamp-2 text-[12px] leading-relaxed text-ink-faint">
                  {suggestion.prompt}
                </p>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
