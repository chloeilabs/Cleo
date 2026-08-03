import type { ReactNode } from "react"

import { cn } from "@/lib/utils"

interface SectionHeadingProps {
  eyebrow: string
  title: ReactNode
  description: ReactNode
  inverted?: boolean
  className?: string
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  inverted = false,
  className,
}: SectionHeadingProps) {
  return (
    <div className={cn("max-w-2xl", className)}>
      <p
        className={cn(
          "mb-3 text-xs tracking-[0.04em] uppercase",
          inverted ? "text-strong-foreground/50" : "text-muted-foreground",
        )}
      >
        {eyebrow}
      </p>
      <h2
        className={cn(
          "text-[clamp(1.65rem,3.2vw,2.35rem)] font-semibold leading-[1.12] tracking-[-0.035em]",
          inverted ? "text-strong-foreground" : "text-foreground",
        )}
      >
        {title}
      </h2>
      <div
        className={cn(
          "mt-3 text-sm leading-6",
          inverted ? "text-strong-foreground/60" : "text-muted-foreground",
        )}
      >
        {description}
      </div>
    </div>
  )
}
