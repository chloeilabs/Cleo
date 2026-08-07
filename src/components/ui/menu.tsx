"use client";

import {
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";

import { classNames } from "@/lib/format";

interface MenuProps {
  /** Rendered inside the trigger button. */
  trigger: ReactNode;
  /** Receives a `close` callback so items can dismiss the menu. */
  children: (close: () => void) => ReactNode;
  align?: "start" | "end";
  side?: "top" | "bottom";
  className?: string;
  triggerClassName?: string;
  panelClassName?: string;
  label?: string;
  disabled?: boolean;
}

/**
 * A small popover menu. Deliberately unmanaged by any library: it closes on
 * outside pointer-down and Escape, restores focus to the trigger, and lets the
 * caller lay out the panel contents however it likes.
 */
export function Menu({
  trigger,
  children,
  align = "start",
  side = "bottom",
  className,
  triggerClassName,
  panelClassName,
  label,
  disabled,
}: MenuProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelId = useId();

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;

    const container = containerRef.current;
    const triggerNode = triggerRef.current;

    const onPointerDown = (event: PointerEvent) => {
      if (!container?.contains(event.target as Node)) setOpen(false);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        setOpen(false);
      }
    };

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown, true);

    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown, true);

      // Hand focus back to the trigger, but only when it was still inside the
      // menu — an outside click should keep focus wherever the user put it.
      if (container?.contains(document.activeElement)) triggerNode?.focus();
    };
  }, [open]);

  return (
    <div ref={containerRef} className={classNames("relative", className)}>
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        aria-label={label}
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
        className={classNames(
          "inline-flex items-center gap-1.5 rounded-md transition-colors disabled:cursor-not-allowed disabled:opacity-50",
          triggerClassName,
        )}
      >
        {trigger}
      </button>

      {open ? (
        <div
          id={panelId}
          role="menu"
          className={classNames(
            "animate-fade-up absolute z-50 min-w-56 overflow-hidden rounded-lg border border-hairline bg-overlay p-1 shadow-2xl shadow-black/60",
            side === "bottom" ? "top-[calc(100%+6px)]" : "bottom-[calc(100%+6px)]",
            align === "start" ? "left-0" : "right-0",
            panelClassName,
          )}
        >
          {children(close)}
        </div>
      ) : null}
    </div>
  );
}

interface MenuItemProps {
  onSelect: () => void;
  children: ReactNode;
  selected?: boolean;
  destructive?: boolean;
  disabled?: boolean;
  hint?: ReactNode;
}

export function MenuItem({
  onSelect,
  children,
  selected,
  destructive,
  disabled,
  hint,
}: MenuItemProps) {
  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      onClick={onSelect}
      className={classNames(
        "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-[13px] transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-50",
        destructive
          ? "text-negative hover:bg-negative/10"
          : "text-ink-muted hover:bg-raised hover:text-ink",
        selected && !destructive && "bg-raised text-ink",
      )}
    >
      <span className="min-w-0 flex-1 truncate">{children}</span>
      {hint ? (
        <span className="shrink-0 text-[11px] text-ink-faint">{hint}</span>
      ) : null}
    </button>
  );
}

export function MenuLabel({ children }: { children: ReactNode }) {
  return (
    <div className="px-2.5 pt-2 pb-1 text-[10px] font-semibold tracking-[0.08em] text-ink-faint uppercase">
      {children}
    </div>
  );
}

export function MenuSeparator() {
  return <div className="my-1 h-px bg-hairline" />;
}
