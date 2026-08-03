import { ArrowUpRight } from "lucide-react"

import { compactNumber, fixed } from "@/lib/format"
import { navigate } from "@/lib/routing"
import type { ModelProfile } from "@/types"

export function Hero({ profile }: { profile: ModelProfile }) {
  const { identity, architecture, metrics, generalization } = profile

  return (
    <section id="top" className="bg-background pt-28 pb-16 sm:pt-32 sm:pb-20">
      <div className="mx-auto max-w-[720px] px-5 text-center sm:px-8">
        <p className="text-[13px] text-muted-foreground">
          Product Release
          <span className="mx-2 text-border">·</span>
          {identity.release}
        </p>

        <h1 className="mt-6 text-[clamp(2.4rem,5.5vw,3.75rem)] font-semibold leading-[1.08] tracking-[-0.04em]">
          {identity.model_name}: Broader language from a fully inspectable stack
        </h1>

        <p className="mx-auto mt-5 max-w-[540px] text-[15px] leading-7 text-muted-foreground sm:text-base sm:leading-7">
          A from-scratch general-language research model trained on one Apple M4—no pretrained
          checkpoint, no borrowed tokenizer. Continued pretraining, instruction tuning, and gated
          identity repair in a transparent local pipeline.
        </p>
      </div>

      <div className="mx-auto mt-12 max-w-[960px] px-5 sm:px-8">
        <div className="overflow-hidden rounded-[12px] border border-border bg-background">
          <div className="hero-media relative min-h-[300px] overflow-hidden sm:min-h-[400px]">
            <div className="hero-orb absolute inset-0 opacity-90 dark:opacity-100" aria-hidden="true" />
            <div className="relative z-10 grid min-h-[300px] place-items-center px-6 py-20 sm:min-h-[400px]">
              <div className="text-center">
                <p className="text-sm text-foreground/70">Introducing {identity.model_name}</p>
                <p className="mt-4 text-5xl font-semibold tracking-tight sm:text-6xl">
                  {compactNumber(metrics.parameter_count)}
                </p>
                <p className="mt-2 text-sm text-muted-foreground">parameters · from random weights</p>
                <button
                  type="button"
                  onClick={() => navigate("chat")}
                  className="mx-auto mt-8 grid size-14 place-items-center rounded-full bg-foreground text-background shadow-[0_10px_40px_rgba(0,0,0,0.12)] transition-transform hover:scale-[1.04]"
                  aria-label={`Try ${identity.model_name}`}
                >
                  <ArrowUpRight className="size-5" />
                </button>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 border-t border-border bg-background sm:grid-cols-4">
            {[
              [compactNumber(metrics.parameter_count), "Parameters"],
              [`${fixed(generalization.general_loss_reduction_percent, 1)}%`, "Lower general loss"],
              [
                `${fixed(generalization.instruction_loss_reduction_percent, 1)}%`,
                "Lower instruction loss",
              ],
              [`${architecture.block_size}`, "Token context"],
            ].map(([value, label]) => (
              <div
                key={label}
                className="border-border px-4 py-4 text-left sm:border-r sm:last:border-r-0 [&:nth-child(odd)]:border-r max-sm:[&:nth-child(-n+2)]:border-b"
              >
                <div className="text-lg font-semibold tracking-tight">{value}</div>
                <div className="mt-1 text-xs text-muted-foreground">{label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
