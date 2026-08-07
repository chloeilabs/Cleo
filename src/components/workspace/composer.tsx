"use client";

import {
  type ChangeEvent,
  type KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  ArrowUpIcon,
  BranchIcon,
  CheckIcon,
  ChevronDownIcon,
  CloseIcon,
  ImageIcon,
  PullRequestIcon,
  RepoIcon,
  SparkIcon,
  StopIcon,
} from "@/components/icons";
import { Button } from "@/components/ui/button";
import { Menu, MenuItem, MenuLabel, MenuSeparator } from "@/components/ui/menu";
import { useWorkspace } from "@/components/workspace/provider";
import { classNames } from "@/lib/format";
import type { ConversationMode } from "@/lib/types";

export interface ComposerSubmission {
  prompt: string;
  images: Array<{ data: string; mimeType: string }>;
}

interface ComposerProps {
  /** `create` shows repository and PR controls; `follow-up` hides them. */
  variant: "create" | "follow-up";
  placeholder: string;
  busy?: boolean;
  disabled?: boolean;
  /** Shown instead of send when a run is in flight. */
  onCancel?: () => void;
  onSubmit: (submission: ComposerSubmission) => void | Promise<void>;
  autoFocus?: boolean;
}

const MAX_IMAGE_BYTES = 4 * 1024 * 1024;

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

interface Attachment {
  id: string;
  name: string;
  mimeType: string;
  data: string;
  preview: string;
}

export function Composer({
  variant,
  placeholder,
  busy = false,
  disabled = false,
  onCancel,
  onSubmit,
  autoFocus = false,
}: ComposerProps) {
  const {
    models,
    repositories,
    preferences,
    setPreferences,
    selectedModel,
    catalogLoading,
  } = useWorkspace();

  const [prompt, setPrompt] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (autoFocus) textareaRef.current?.focus();
  }, [autoFocus]);

  // Grow with the content, up to roughly ten lines.
  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 240)}px`;
  }, [prompt]);

  const selectedRepository = useMemo(
    () => repositories.find((repo) => repo.url === preferences.repoUrl),
    [repositories, preferences.repoUrl],
  );

  const canSubmit = prompt.trim().length > 0 && !busy && !disabled;

  const submit = async () => {
    if (!canSubmit) return;

    const submission: ComposerSubmission = {
      prompt: prompt.trim(),
      images: attachments.map(({ data, mimeType }) => ({ data, mimeType })),
    };

    setPrompt("");
    setAttachments([]);
    await onSubmit(submission);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    const isSend =
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing;

    if (isSend) {
      event.preventDefault();
      void submit();
    }
  };

  const addFiles = async (files: FileList | null) => {
    if (!files?.length) return;

    const next: Attachment[] = [];
    for (const file of Array.from(files).slice(0, 4)) {
      if (!file.type.startsWith("image/") || file.size > MAX_IMAGE_BYTES) {
        continue;
      }
      const dataUrl = await readAsDataUrl(file);
      next.push({
        id: `${file.name}-${file.lastModified}`,
        name: file.name,
        mimeType: file.type,
        data: dataUrl.split(",")[1] ?? "",
        preview: dataUrl,
      });
    }

    setAttachments((current) => [...current, ...next].slice(0, 4));
  };

  const onPaste = (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const files = event.clipboardData?.files;
    if (files?.length) void addFiles(files);
  };

  const modeLabel = preferences.mode === "plan" ? "Plan" : "Agent";

  return (
    <div
      className={classNames(
        "rounded-xl border bg-panel transition-colors",
        disabled ? "border-hairline opacity-60" : "border-hairline-strong",
        "focus-within:border-accent/60",
      )}
    >
      {attachments.length > 0 ? (
        <div className="flex flex-wrap gap-2 border-b border-hairline px-3 py-2.5">
          {attachments.map((attachment) => (
            <div
              key={attachment.id}
              className="group relative size-14 overflow-hidden rounded-md border border-hairline"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={attachment.preview}
                alt={attachment.name}
                className="size-full object-cover"
              />
              <button
                type="button"
                aria-label={`Remove ${attachment.name}`}
                onClick={() =>
                  setAttachments((current) =>
                    current.filter((entry) => entry.id !== attachment.id),
                  )
                }
                className="absolute top-0.5 right-0.5 rounded bg-canvas/85 p-0.5 text-ink-muted opacity-0 transition-opacity group-hover:opacity-100 hover:text-ink"
              >
                <CloseIcon className="size-3" />
              </button>
            </div>
          ))}
        </div>
      ) : null}

      <textarea
        ref={textareaRef}
        value={prompt}
        rows={variant === "create" ? 3 : 2}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
          setPrompt(event.target.value)
        }
        onKeyDown={onKeyDown}
        onPaste={onPaste}
        className="block w-full resize-none bg-transparent px-3.5 pt-3 pb-2 text-[14px] leading-[1.6] text-ink outline-none disabled:cursor-not-allowed"
      />

      <div className="flex flex-wrap items-center gap-1.5 px-2.5 pt-1 pb-2.5">
        <Menu
          label="Select model"
          disabled={catalogLoading && models.length === 0}
          triggerClassName="h-7 gap-1.5 px-2 text-[12px] text-ink-muted hover:bg-raised hover:text-ink"
          panelClassName="max-h-80 overflow-y-auto"
          side="top"
          trigger={
            <>
              <SparkIcon className="size-3.5" />
              <span className="max-w-40 truncate">
                {selectedModel?.label ?? "Model"}
              </span>
              <ChevronDownIcon className="size-3" />
            </>
          }
        >
          {(close) => (
            <>
              <MenuLabel>Model</MenuLabel>
              {models.length === 0 ? (
                <p className="px-2.5 py-2 text-[12px] text-ink-faint">
                  No models available for this API key.
                </p>
              ) : (
                models.map((model) => (
                  <MenuItem
                    key={model.key}
                    selected={model.key === selectedModel?.key}
                    hint={
                      model.key === selectedModel?.key ? (
                        <CheckIcon className="size-3.5 text-accent" />
                      ) : undefined
                    }
                    onSelect={() => {
                      setPreferences({ ...preferences, modelKey: model.key });
                      close();
                    }}
                  >
                    {model.label}
                  </MenuItem>
                ))
              )}
            </>
          )}
        </Menu>

        <div className="flex h-7 items-center rounded-md border border-hairline p-0.5">
          {(["agent", "plan"] as ConversationMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setPreferences({ ...preferences, mode })}
              className={classNames(
                "rounded px-2 py-0.5 text-[11px] font-medium capitalize transition-colors",
                preferences.mode === mode
                  ? "bg-raised text-ink"
                  : "text-ink-faint hover:text-ink-muted",
              )}
            >
              {mode}
            </button>
          ))}
        </div>

        {variant === "create" ? (
          <>
            <Menu
              label="Select repository"
              triggerClassName="h-7 gap-1.5 px-2 text-[12px] text-ink-muted hover:bg-raised hover:text-ink"
              panelClassName="max-h-80 w-72 overflow-y-auto"
              side="top"
              trigger={
                <>
                  <RepoIcon className="size-3.5" />
                  <span className="max-w-48 truncate">
                    {selectedRepository?.label ?? "No repository"}
                  </span>
                  <ChevronDownIcon className="size-3" />
                </>
              }
            >
              {(close) => (
                <>
                  <MenuLabel>Repository</MenuLabel>
                  <MenuItem
                    selected={!preferences.repoUrl}
                    onSelect={() => {
                      setPreferences({ ...preferences, repoUrl: undefined });
                      close();
                    }}
                  >
                    No repository — empty workspace
                  </MenuItem>

                  {repositories.length > 0 ? <MenuSeparator /> : null}

                  {repositories.map((repository) => (
                    <MenuItem
                      key={repository.url}
                      selected={repository.url === preferences.repoUrl}
                      onSelect={() => {
                        setPreferences({
                          ...preferences,
                          repoUrl: repository.url,
                        });
                        close();
                      }}
                    >
                      {repository.label}
                    </MenuItem>
                  ))}

                  {repositories.length === 0 && !catalogLoading ? (
                    <p className="px-2.5 py-2 text-[12px] leading-relaxed text-ink-faint">
                      No repositories connected. Link GitHub to your Cursor team
                      to dispatch agents against a repo.
                    </p>
                  ) : null}
                </>
              )}
            </Menu>

            {preferences.repoUrl ? (
              <label className="flex h-7 items-center gap-1.5 rounded-md px-2 text-[12px] text-ink-muted transition-colors focus-within:bg-raised hover:bg-raised">
                <BranchIcon className="size-3.5 shrink-0" />
                <input
                  value={preferences.startingRef}
                  placeholder="default branch"
                  onChange={(event) =>
                    setPreferences({
                      ...preferences,
                      startingRef: event.target.value,
                    })
                  }
                  className="w-28 bg-transparent text-[12px] text-ink outline-none"
                />
              </label>
            ) : null}

            {preferences.repoUrl ? (
              <button
                type="button"
                onClick={() =>
                  setPreferences({
                    ...preferences,
                    autoCreatePR: !preferences.autoCreatePR,
                  })
                }
                className={classNames(
                  "inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[12px] transition-colors",
                  preferences.autoCreatePR
                    ? "bg-accent-soft text-accent"
                    : "text-ink-muted hover:bg-raised hover:text-ink",
                )}
                title="Open a pull request when the run finishes"
              >
                <PullRequestIcon className="size-3.5" />
                Auto PR
              </button>
            ) : null}
          </>
        ) : null}

        <div className="ml-auto flex items-center gap-1.5">
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(event) => {
              void addFiles(event.target.files);
              event.target.value = "";
            }}
          />
          <Button
            variant="ghost"
            size="icon"
            aria-label="Attach an image"
            onClick={() => fileRef.current?.click()}
            disabled={disabled}
          >
            <ImageIcon className="size-4" />
          </Button>

          {busy && onCancel ? (
            <Button
              variant="secondary"
              size="sm"
              onClick={onCancel}
              className="gap-1.5"
            >
              <StopIcon className="size-3" />
              Stop
            </Button>
          ) : (
            <Button
              variant="primary"
              size="icon"
              aria-label="Send"
              disabled={!canSubmit}
              onClick={() => void submit()}
              className="size-7"
            >
              <ArrowUpIcon className="size-4" strokeWidth={2} />
            </Button>
          )}
        </div>
      </div>

      <p className="px-3.5 pb-2.5 text-[11px] text-ink-faint">
        {modeLabel} mode ·{" "}
        <kbd className="font-sans">Enter</kbd> to send ·{" "}
        <kbd className="font-sans">Shift + Enter</kbd> for a new line
      </p>
    </div>
  );
}
