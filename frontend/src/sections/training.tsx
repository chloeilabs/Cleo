import { BadgeCheck, CheckCircle2, Database, Dumbbell, Fingerprint, MessagesSquare } from "lucide-react"

import { fixed, number } from "@/lib/format"
import type { ModelProfile } from "@/types"

export function Training({ profile }: { profile: ModelProfile }) {
  const { training, dataset, generalization, adaptation } = profile
  const timeline = [
    {
      icon: Database,
      step: "FOUNDATION 01",
      title: `${number(generalization.foundation_steps)} steps from random weights`,
      copy: `The original language foundation used ${number(dataset.train_stories)} TinyStories examples. That provenance remains part of this release, but it is no longer the model's only training distribution.`,
    },
    {
      icon: Fingerprint,
      step: "TOKEN 02",
      title: "Tokenizer learned locally",
      copy: "Deterministic byte-level BPE trained from an 8 MiB corpus sample. All UTF-8 text round-trips without unknown tokens.",
    },
    {
      icon: Dumbbell,
      step: "GENERALIZE 03",
      title: `${number(generalization.continued_pretraining_steps)} continued-pretraining steps`,
      copy: `${number(dataset.general.train_documents)} WikiText documents and ${number(dataset.general.train_tokens)} prepared tokens broadened the language distribution. Ten percent of microbatches retained foundation data.`,
    },
    {
      icon: MessagesSquare,
      step: "INSTRUCT 04",
      title: `${number(generalization.instruction_tuning_steps)} instruction-tuning steps`,
      copy: `${number(dataset.instruction.train_examples)} Dolly examples across eight categories trained answer-only targets, mixed with general-language, foundation, and identity-retention batches.`,
    },
    ...(generalization.accepted && adaptation.identity_tuned
      ? [{
          icon: BadgeCheck,
          step: "REPAIR 05",
          title: `${number(generalization.identity_repair_steps)}-step identity repair`,
          copy: `${fixed(adaptation.held_out_exact_match * 100, 0)}% exact match on eight held-out identity prompts while preserving general, instruction, and foundation losses inside the repair gates.`,
        }]
      : []),
    {
      icon: CheckCircle2,
      step: "RELEASE 06",
      title: "Self-describing gated checkpoint",
      copy: "Weights, configuration, RNG state, tokenizer checksum, source manifests, stage metrics, and acceptance results travel with the release.",
    },
  ]

  return (
    <section id="training" className="py-28 sm:py-32">
      <div className="mx-auto grid max-w-[1240px] gap-14 px-5 sm:px-8 lg:grid-cols-[.9fr_1.1fr] lg:gap-20">
        <div className="lg:sticky lg:top-28 lg:self-start">
          <div className="mb-5 flex items-center gap-2.5 text-[11px] font-bold uppercase tracking-[.18em] text-brand-green">
            <span className="size-1.5 rounded-full bg-current" /> Training run
          </div>
          <h2 className="max-w-lg text-6xl font-semibold leading-[.92] tracking-[-.06em] sm:text-7xl">
            Ground up means ground up.
          </h2>
          <p className="mt-7 max-w-lg text-base leading-7 text-muted-foreground">
            Random initialization to the promoted alpha checkpoint in {training.duration} on Apple Silicon. The autoregressive stages processed {number(training.tokens_seen)} token presentations; every later stage had explicit retention gates.
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
