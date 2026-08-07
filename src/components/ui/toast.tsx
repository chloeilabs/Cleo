"use client";

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";

import { CloseIcon, ExternalIcon } from "@/components/icons";
import { classNames } from "@/lib/format";

interface Toast {
  id: number;
  tone: "error" | "info" | "success";
  title: string;
  detail?: string;
  helpUrl?: string;
}

interface ToastApi {
  notify: (toast: Omit<Toast, "id">) => void;
  reportError: (error: unknown, fallback?: string) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

const TONE_STYLES: Record<Toast["tone"], string> = {
  error: "border-negative/35 bg-negative/10",
  info: "border-hairline bg-overlay",
  success: "border-positive/30 bg-positive/10",
};

const TITLE_STYLES: Record<Toast["tone"], string> = {
  error: "text-negative",
  info: "text-ink",
  success: "text-positive",
};

let nextId = 1;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const notify = useCallback(
    (toast: Omit<Toast, "id">) => {
      const id = nextId++;
      setToasts((current) => [...current.slice(-3), { ...toast, id }]);
      setTimeout(() => dismiss(id), toast.tone === "error" ? 9000 : 4500);
    },
    [dismiss],
  );

  const reportError = useCallback(
    (error: unknown, fallback = "Something went wrong.") => {
      const message = error instanceof Error ? error.message : fallback;
      const helpUrl =
        error && typeof error === "object" && "helpUrl" in error
          ? (error as { helpUrl?: string }).helpUrl
          : undefined;
      notify({ tone: "error", title: message, helpUrl });
    },
    [notify],
  );

  const api = useMemo(() => ({ notify, reportError }), [notify, reportError]);

  return (
    <ToastContext.Provider value={api}>
      {children}

      <div className="pointer-events-none fixed bottom-4 left-1/2 z-100 flex w-full max-w-md -translate-x-1/2 flex-col gap-2 px-4">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            role="status"
            className={classNames(
              "animate-fade-up pointer-events-auto flex items-start gap-3 rounded-lg border px-3.5 py-3 shadow-xl shadow-black/50 backdrop-blur",
              TONE_STYLES[toast.tone],
            )}
          >
            <div className="min-w-0 flex-1">
              <p
                className={classNames(
                  "text-[13px] leading-snug font-medium",
                  TITLE_STYLES[toast.tone],
                )}
              >
                {toast.title}
              </p>
              {toast.detail ? (
                <p className="mt-0.5 text-xs text-ink-muted">{toast.detail}</p>
              ) : null}
              {toast.helpUrl ? (
                <a
                  href={toast.helpUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-1.5 inline-flex items-center gap-1 text-xs text-accent hover:underline"
                >
                  Open Cursor settings
                  <ExternalIcon className="size-3" />
                </a>
              ) : null}
            </div>

            <button
              type="button"
              aria-label="Dismiss"
              onClick={() => dismiss(toast.id)}
              className="shrink-0 rounded p-0.5 text-ink-faint transition-colors hover:text-ink"
            >
              <CloseIcon className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used inside a ToastProvider.");
  }
  return context;
}
