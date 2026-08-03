import { SectionHeading } from "@/components/section-heading"
import { Progress } from "@/components/ui/progress"
import { fixed } from "@/lib/format"
import type { ModelProfile } from "@/types"

function LossRow({
  label,
  baseline,
  release,
  reduction,
  note,
}: {
  label: string
  baseline: number
  release: number
  reduction: number
  note: string
}) {
  return (
    <div className="border-t border-border py-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-sm font-medium">{label}</p>
          <p className="mt-1 text-xs text-muted-foreground">{note}</p>
        </div>
        <p className="text-3xl font-semibold tracking-tight">{fixed(reduction, 1)}%</p>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-6 text-sm">
        <div>
          <span className="text-muted-foreground">Before</span>
          <strong className="mt-1 block font-medium tabular-nums">{fixed(baseline, 4)}</strong>
        </div>
        <div>
          <span className="text-muted-foreground">Release</span>
          <strong className="mt-1 block font-medium tabular-nums">{fixed(release, 4)}</strong>
        </div>
      </div>
      <Progress value={Math.min(reduction, 100)} className="mt-4 h-1" />
    </div>
  )
}

export function Benchmarks({ profile }: { profile: ModelProfile }) {
  const { generalization, benchmark } = profile
  const storyChange = (generalization.story_retention_ratio - 1) * 100

  return (
    <section id="benchmarks" className="border-t border-border py-16 sm:py-20">
      <div className="mx-auto max-w-5xl px-5 sm:px-8">
        <SectionHeading
          eyebrow="Benchmarks"
          title="Broader, by measured gates."
          description={
            <p>
              Same tokenizer and fixed held-out batches before and after generalization. Internal
              fit metrics—not a public leaderboard claim.
            </p>
          }
        />

        <div className="mt-8">
          <LossRow
            label="WikiText general-language validation"
            baseline={generalization.general_baseline_loss}
            release={generalization.general_validation_loss}
            reduction={generalization.general_loss_reduction_percent}
            note={`Perplexity ${fixed(generalization.general_validation_perplexity, 2)} · lower is better`}
          />
          <LossRow
            label="Dolly instruction validation"
            baseline={generalization.instruction_baseline_loss}
            release={generalization.instruction_validation_loss}
            reduction={generalization.instruction_loss_reduction_percent}
            note="Answer-token cross-entropy · lower is better"
          />
        </div>

        <div className="mt-2 grid grid-cols-1 border-y border-border sm:grid-cols-3">
          {[
            [`${fixed(benchmark.cached_tokens_per_second, 1)} tok/s`, "KV-cached"],
            [`${fixed(benchmark.uncached_tokens_per_second, 1)} tok/s`, "Full-forward"],
            [`${fixed(benchmark.cache_speedup, 2)}×`, "Cache speedup"],
          ].map(([value, label]) => (
            <div key={label} className="border-border py-5 sm:border-r sm:px-5 sm:first:pl-0 sm:last:border-r-0 sm:last:pr-0">
              <strong className="text-xl font-semibold tracking-tight">{value}</strong>
              <span className="mt-1 block text-xs text-muted-foreground">{label}</span>
            </div>
          ))}
        </div>

        <div className="mt-8 grid gap-6 border-t border-border pt-6 md:grid-cols-2">
          <div>
            <p className="text-xs tracking-[0.04em] text-muted-foreground uppercase">Identity gate</p>
            <p className="mt-2 text-3xl font-semibold tracking-tight">8 / 8</p>
            <p className="mt-1 text-sm text-muted-foreground">held-out prompts matched exactly</p>
          </div>
          <div>
            <p className="text-xs tracking-[0.04em] text-muted-foreground uppercase">Retention cost</p>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              Foundation loss is{" "}
              <span className="font-medium text-foreground">{fixed(storyChange, 1)}% higher</span>{" "}
              than the pre-generalization release—under the 35% ceiling.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
