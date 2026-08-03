import type { ReactNode } from "react"

interface SectionHeadingProps {
  eyebrow: string
  title: ReactNode
  description: ReactNode
  inverted?: boolean
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  inverted = false,
}: SectionHeadingProps) {
  return (
    <div className="grid items-end gap-8 lg:grid-cols-[1.05fr_.95fr]">
      <div>
        <div
          className={`mb-5 flex items-center gap-2.5 text-[11px] font-bold uppercase tracking-[0.18em] ${
            inverted ? "text-brand-lime" : "text-brand-green"
          }`}
        >
          <span className="size-1.5 rounded-full bg-current" />
          {eyebrow}
        </div>
        <h2
          className={`max-w-3xl text-5xl font-semibold leading-[.94] tracking-[-0.055em] sm:text-6xl lg:text-7xl ${
            inverted ? "text-white" : "text-foreground"
          }`}
        >
          {title}
        </h2>
      </div>
      <div
        className={`max-w-xl text-base leading-7 ${
          inverted ? "text-white/60" : "text-muted-foreground"
        }`}
      >
        {description}
      </div>
    </div>
  )
}
