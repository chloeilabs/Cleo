import { ArrowDownRight, CheckCircle2, Scale } from "lucide-react"

import { SectionHeading } from "@/components/section-heading"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { fixed } from "@/lib/format"
import type { ModelProfile } from "@/types"

interface ComparisonProps {
  eyebrow: string
  baseline: number
  release: number
  reduction: number
  note: string
}

function LossComparison({ eyebrow, baseline, release, reduction, note }: ComparisonProps) {
  return (
    <Card className="rounded-3xl bg-card py-0 shadow-none">
      <CardHeader className="p-7 pb-4">
        <div className="text-[10px] font-bold uppercase tracking-[.14em] text-muted-foreground">{eyebrow}</div>
        <div className="mt-4 flex items-end gap-3">
          <strong className="text-6xl font-semibold tracking-[-.065em]">{fixed(reduction, 1)}%</strong>
          <span className="mb-2 flex items-center gap-1 text-sm font-medium text-brand-green"><ArrowDownRight className="size-4" /> lower loss</span>
        </div>
      </CardHeader>
      <CardContent className="p-7 pt-2">
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-2xl bg-muted/65 p-4">
            <span className="text-[10px] font-bold uppercase tracking-[.1em] text-muted-foreground">Before</span>
            <strong className="mt-2 block text-2xl">{fixed(baseline, 4)}</strong>
          </div>
          <div className="rounded-2xl bg-brand-lime/35 p-4">
            <span className="text-[10px] font-bold uppercase tracking-[.1em] text-brand-green">Release</span>
            <strong className="mt-2 block text-2xl">{fixed(release, 4)}</strong>
          </div>
        </div>
        <Progress value={reduction} className="mt-5 h-2 bg-muted **:data-[slot=progress-indicator]:bg-brand-green" />
        <p className="mt-4 text-xs leading-5 text-muted-foreground">{note}</p>
      </CardContent>
    </Card>
  )
}

export function Benchmarks({ profile }: { profile: ModelProfile }) {
  const { generalization, benchmark } = profile
  const storyChange = (generalization.story_retention_ratio - 1) * 100

  return (
    <section id="benchmarks" className="py-28 sm:py-32">
      <div className="mx-auto max-w-[1240px] px-5 sm:px-8">
        <SectionHeading
          eyebrow="Evaluation"
          title="Broader, by measured gates."
          description={
            <p>
              We compare the same checkpoint before and after generalization with the same tokenizer and fixed held-out batches. These are internal distribution-fit measurements—not a public leaderboard or a claim of frontier-level intelligence.
            </p>
          }
        />
        <div className="mt-14 grid gap-5 lg:grid-cols-2">
          <LossComparison
            eyebrow="WikiText general-language validation"
            baseline={generalization.general_baseline_loss}
            release={generalization.general_validation_loss}
            reduction={generalization.general_loss_reduction_percent}
            note={`Release perplexity ${fixed(generalization.general_validation_perplexity, 2)} · fixed validation batches · lower is better`}
          />
          <LossComparison
            eyebrow="Dolly instruction validation"
            baseline={generalization.instruction_baseline_loss}
            release={generalization.instruction_validation_loss}
            reduction={generalization.instruction_loss_reduction_percent}
            note="Answer-token cross-entropy on a deterministic, category-stratified held-out split · lower is better"
          />
        </div>

        <div className="mt-5 grid overflow-hidden rounded-3xl border bg-card sm:grid-cols-3">
          {[
            [`${fixed(benchmark.cached_tokens_per_second, 1)} tok/s`, "KV-cached median"],
            [`${fixed(benchmark.uncached_tokens_per_second, 1)} tok/s`, "full-forward median"],
            [`${fixed(benchmark.cache_speedup, 2)}×`, "cache speedup"],
          ].map(([value, label], index) => (
            <div key={label} className={`p-6 ${index < 2 ? "border-b sm:border-b-0 sm:border-r" : ""}`}>
              <strong className="text-3xl font-semibold tracking-[-.045em]">{value}</strong>
              <span className="mt-2 block text-xs text-muted-foreground">{label}</span>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs leading-5 text-muted-foreground">{benchmark.device} · FP32 · batch 1 · {benchmark.new_tokens} generated tokens · five-run warmed median · {benchmark.outputs_equal ? "cached and uncached outputs matched" : "output parity not established"}</p>

        <Card className="mt-5 rounded-3xl bg-brand-ink py-0 text-white shadow-none">
          <CardContent className="grid gap-8 p-7 md:grid-cols-[.8fr_1.2fr] md:p-9">
            <div>
              <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[.14em] text-brand-lime">
                <CheckCircle2 className="size-4" /> Promotion gates passed
              </div>
              <strong className="mt-4 block text-5xl font-semibold tracking-[-.06em]">8 / 8</strong>
              <p className="mt-2 text-sm text-white/55">held-out identity prompts matched exactly</p>
            </div>
            <div className="border-white/10 md:border-l md:pl-9">
              <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[.14em] text-white/45"><Scale className="size-4" /> Retention cost</div>
              <p className="mt-4 text-lg leading-8 text-white/80">
                Foundation-distribution loss is <strong className="text-white">{fixed(storyChange, 1)}% higher</strong> than the pre-generalization release. That stayed below the declared 35% ceiling, but it is a real tradeoff—not a free capability gain.
              </p>
              <div className="mt-5 flex flex-wrap gap-2">
                <Badge className="rounded-full border-white/10 bg-white/8 text-white">General gate ≤ 0.80×</Badge>
                <Badge className="rounded-full border-white/10 bg-white/8 text-white">Instruction gate ≤ 0.65×</Badge>
                <Badge className="rounded-full border-white/10 bg-white/8 text-white">Identity gate = 100%</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="mt-5 rounded-2xl border bg-white/35 p-5 text-sm leading-6 text-muted-foreground">
          <strong className="text-foreground">Important:</strong> lower loss does not prove reliable reasoning, factuality, coding, arithmetic, or safety. Qualitative probes still show repetition and incorrect answers, so this release is labeled an experimental general-language alpha rather than a general-purpose assistant.
        </div>
      </div>
    </section>
  )
}
