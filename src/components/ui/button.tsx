"use client";

import type { ButtonHTMLAttributes, ReactNode } from "react";

import { SpinnerIcon } from "@/components/icons";
import { classNames } from "@/lib/format";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "icon";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-accent text-white hover:bg-accent-strong disabled:bg-accent/40 disabled:text-white/70",
  secondary:
    "bg-raised text-ink border border-hairline hover:border-hairline-strong hover:bg-overlay",
  ghost: "text-ink-muted hover:text-ink hover:bg-raised",
  danger:
    "text-negative border border-negative/30 hover:bg-negative/10 hover:border-negative/50",
};

const SIZES: Record<Size, string> = {
  sm: "h-7 gap-1.5 px-2.5 text-xs",
  md: "h-9 gap-2 px-3.5 text-sm",
  icon: "h-7 w-7 justify-center",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  children?: ReactNode;
}

export function Button({
  variant = "secondary",
  size = "md",
  loading = false,
  disabled,
  className,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled || loading}
      className={classNames(
        "inline-flex shrink-0 items-center rounded-md font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-60",
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...props}
    >
      {loading ? (
        <SpinnerIcon className="size-3.5 animate-spin-slow" />
      ) : null}
      {children}
    </button>
  );
}
