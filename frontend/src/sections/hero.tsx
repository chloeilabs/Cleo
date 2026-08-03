import { ArrowDownRight, BookOpen, Cpu, Layers3 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { compactNumber, fixed, number } from "@/lib/format"
import type { ModelProfile } from "@/types"

export function Hero({ profile }: { profile: ModelProfile }) {
  const { identity, runtime, architecture, metrics, benchmark } = profile
  const headlineMetrics = [
    [compactNumber(metrics.parameter_count), "trainable parameters"],
    [fixed(metrics.best_validation_loss, 3), "best validation loss"],
    [fixed(metrics.best_validation_perplexity, 2), "validation perplexity"],
    [`${fixed(benchmark.cache_speedup, 2)}×`, "KV-cache speedup"],
  ]
  const signalSpecs = [
    { icon: Cpu, label: "Context", value: `${architecture.block_size} tokens` },
    { icon: Layers3, label: "Layers", value: `${architecture.n_layer} blocks` },
    { icon: BookOpen, label: "Vocab", value: `${number(architecture.vocab_size)} BPE` },
  ]

  return (
    <>
      <section
        id="top"
        className="relative overflow-hidden bg-brand-ink pb-28 pt-32 text-white sm:pt-36 lg:pb-32"
      >
        <div className="hero-grid absolute inset-0 opacity-25" aria-hidden="true" />
        <div className="relative mx-auto grid max-w-[1320px] items-center gap-14 px-5 sm:px-8 lg:grid-cols-[1.08fr_.92fr] lg:gap-20">
          <div>
            <div className="mb-6 flex items-center gap-2.5 text-[11px] font-bold uppercase tracking-[0.18em] text-brand-lime">
              <span className="size-1.5 rounded-full bg-current" />
              {identity.company_name} / {identity.release}
            </div>
            <h1 className="max-w-[760px] text-[clamp(4.5rem,8.2vw,8rem)] font-semibold leading-[.82] tracking-[-0.07em]">
              A small model with a <span className="text-brand-lime">story</span> to tell.
            </h1>
            <p className="mt-8 max-w-2xl text-base leading-7 text-white/62 sm:text-lg">
              {identity.model_name} is fully inspectable and trained from random weights on one
              Apple M4. No pretrained model. No borrowed tokenizer. Just {number(metrics.parameter_count)} learned
              parameters and a very specific job.
            </p>
            <div className="mt-9 flex flex-wrap gap-3">
              <Button
                asChild
                size="lg"
                className="h-12 rounded-full bg-brand-lime px-6 text-brand-ink hover:bg-brand-lime/85"
              >
                <a href="#playground">
                  Open playground <ArrowDownRight />
                </a>
              </Button>
              <Button
                asChild
                variant="outline"
                size="lg"
                className="h-12 rounded-full border-white/20 bg-white/5 px-6 text-white hover:bg-white/10 hover:text-white"
              >
                <a href="#model-card">
                  <BookOpen /> Read model card
                </a>
              </Button>
            </div>
            <p className="mt-7 font-mono text-[11px] uppercase tracking-[0.08em] text-white/38">
              Local research model · decoder-only transformer · {runtime.checkpoint}
            </p>
          </div>

          <div className="mx-auto w-full max-w-[540px]">
            <div className="rounded-[2rem] border border-white/12 bg-white/[0.055] p-5 shadow-[0_40px_100px_rgba(0,0,0,.28)] backdrop-blur sm:p-7">
              <div className="flex items-center justify-between gap-4 font-mono text-[10px] uppercase tracking-[0.1em] text-white/44">
                <span>Model / {identity.model_id.toUpperCase()}</span>
                <Badge className="rounded-full border-brand-lime/25 bg-brand-lime/10 px-3 py-1 text-[10px] text-brand-lime">
                  <span className="mr-1.5 size-1.5 animate-pulse rounded-full bg-current" />
                  Running {runtime.device}
                </Badge>
              </div>
              <div className="relative mx-auto my-8 aspect-square max-w-[350px]">
                <div className="orbit-ring absolute inset-3 rounded-full border border-white/15" />
                <div className="orbit-ring orbit-reverse absolute inset-[16%] rounded-full border border-white/10" />
                <div className="absolute inset-[31%] grid place-items-center rounded-full bg-brand-lime text-center text-brand-ink shadow-[0_0_80px_rgba(202,255,69,.18)]">
                  <div>
                    <strong className="block text-5xl font-semibold tracking-[-0.06em]">8M</strong>
                    <span className="text-[10px] font-bold uppercase tracking-[0.12em] opacity-60">
                      parameters
                    </span>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-3 divide-x divide-white/10 rounded-2xl border border-white/10 bg-black/10 py-4">
                {signalSpecs.map(({ icon: Icon, label, value }) => (
                  <div key={label} className="px-3 text-center">
                    <Icon className="mx-auto mb-2 size-4 text-brand-lime" />
                    <span className="block text-[9px] uppercase tracking-[0.1em] text-white/35">{label}</span>
                    <strong className="mt-1 block text-xs font-medium text-white/85">{value}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="relative z-10 mx-auto -mt-10 grid max-w-[1240px] overflow-hidden rounded-3xl border bg-card shadow-[0_24px_80px_rgba(16,21,17,.1)] sm:grid-cols-2 lg:grid-cols-4">
        {headlineMetrics.map(([value, label], index) => (
          <div
            key={label}
            className={`p-6 sm:p-7 ${index < headlineMetrics.length - 1 ? "border-b sm:border-r lg:border-b-0" : ""}`}
          >
            <strong className="block text-4xl font-semibold tracking-[-0.055em]">{value}</strong>
            <span className="mt-2 block text-xs font-medium text-muted-foreground">{label}</span>
          </div>
        ))}
      </div>
    </>
  )
}
