import { BadgeCheck, CheckCircle2, Database, Dumbbell, Fingerprint } from "lucide-react"

import { fixed, number } from "@/lib/format"
import type { ModelProfile } from "@/types"

export function Training({ profile }: { profile: ModelProfile }) {
  const { training, dataset, metrics, adaptation } = profile
  const timeline = [
    {
      icon: Database,
      step: "DATA 01",
      title: `${number(dataset.train_stories)} training stories`,
      copy: `First official TinyStories training shard plus the complete ${number(dataset.validation_stories)}-story validation split, revision ${dataset.revision.slice(0, 12)}…`,
    },
    {
      icon: Fingerprint,
      step: "TOKEN 02",
      title: "Tokenizer learned locally",
      copy: "Deterministic byte-level BPE trained from an 8 MiB corpus sample. All UTF-8 text round-trips without unknown tokens.",
    },
    {
      icon: Dumbbell,
      step: "TRAIN 03",
      title: `${number(metrics.training_step)} AdamW steps`,
      copy: "8,192 effective tokens per step, 500-step warmup, cosine decay, gradient clipping, and FP32 MPS execution.",
    },
    {
      icon: CheckCircle2,
      step: "CHECK 04",
      title: "Self-describing checkpoint",
      copy: "Weights, optimizer, scheduler, RNG state, model configuration, tokenizer checksum, dataset manifest, and best loss travel together.",
    },
    ...(adaptation.identity_tuned
      ? [{
          icon: BadgeCheck,
          step: "IDENTITY 05",
          title: `${number(adaptation.completed_steps)}-step identity tune`,
          copy: `${fixed(adaptation.held_out_exact_match * 100, 0)}% exact match on held-out identity prompts. Story validation loss changed by ${fixed((adaptation.story_loss_ratio - 1) * 100, 2)}%, inside the 3% retention gate.`,
        }]
      : []),
  ]

  return (
    <section id="training" className="scroll-mt-20 py-28 sm:py-32">
      <div className="mx-auto grid max-w-[1240px] gap-14 px-5 sm:px-8 lg:grid-cols-[.9fr_1.1fr] lg:gap-20">
        <div className="lg:sticky lg:top-28 lg:self-start">
          <div className="mb-5 flex items-center gap-2.5 text-[11px] font-bold uppercase tracking-[.18em] text-brand-green">
            <span className="size-1.5 rounded-full bg-current" /> Training run
          </div>
          <h2 className="max-w-lg text-6xl font-semibold leading-[.92] tracking-[-.06em] sm:text-7xl">
            Ground up means ground up.
          </h2>
          <p className="mt-7 max-w-lg text-base leading-7 text-muted-foreground">
            Random initialization to final checkpoint in {training.duration} on Apple Silicon. The run processed {number(training.tokens_seen)} training-token presentations and improved validation loss at every scheduled evaluation.
          </p>
        </div>
        <div className="border-t">
          {timeline.map(({ icon: Icon, step, title, copy }) => (
            <article key={step} className="grid gap-4 border-b py-7 sm:grid-cols-[110px_1fr] sm:gap-6">
              <div className="flex items-center gap-2 font-mono text-[10px] font-bold tracking-[.1em] text-brand-green">
                <Icon className="size-4" /> {step}
              </div>
              <div>
                <h3 className="text-xl font-semibold tracking-[-.03em]">{title}</h3>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">{copy}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
