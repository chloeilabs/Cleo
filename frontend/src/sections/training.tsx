import { fixed, number } from "@/lib/format"
import type { ModelProfile } from "@/types"

export function Training({ profile }: { profile: ModelProfile }) {
  const { training, dataset, generalization, adaptation } = profile
  const timeline = [
    [
      "Foundation",
      `${number(generalization.foundation_steps)} steps from random weights`,
      `Started on ${number(dataset.train_stories)} TinyStories examples.`,
    ],
    [
      "Tokenizer",
      "Byte-level BPE learned locally",
      "Deterministic merges from an 8 MiB corpus sample.",
    ],
    [
      "Generalize",
      `${number(generalization.continued_pretraining_steps)} continued-pretraining steps`,
      `${number(dataset.general.train_documents)} WikiText documents, with foundation retention.`,
    ],
    [
      "Instruct",
      `${number(generalization.instruction_tuning_steps)} instruction-tuning steps`,
      `${number(dataset.instruction.train_examples)} Dolly examples across eight categories.`,
    ],
    ...(generalization.accepted && adaptation.identity_tuned
      ? ([
          [
            "Repair",
            `${number(generalization.identity_repair_steps)}-step identity repair`,
            `${fixed(adaptation.held_out_exact_match * 100, 0)}% exact match on held-out prompts.`,
          ],
        ] as const)
      : []),
    [
      "Release",
      "Self-describing gated checkpoint",
      "Weights, config, RNG, checksums, and acceptance results travel together.",
    ],
  ] as const

  return (
    <section id="training" className="border-t border-border py-16 sm:py-20">
      <div className="mx-auto grid max-w-5xl gap-10 px-5 sm:px-8 lg:grid-cols-[0.85fr_1.15fr] lg:gap-16">
        <div>
          <p className="text-xs tracking-[0.04em] text-muted-foreground uppercase">Training</p>
          <h2 className="mt-3 text-[clamp(1.65rem,3.2vw,2.35rem)] font-semibold leading-[1.12] tracking-[-0.035em]">
            Ground up means ground up.
          </h2>
          <p className="mt-3 max-w-md text-sm leading-6 text-muted-foreground">
            Random initialization to the promoted alpha in {training.duration}.{" "}
            {number(training.tokens_seen)} token presentations with explicit retention gates.
          </p>
        </div>
        <ol className="divide-y divide-border border-y border-border">
          {timeline.map(([step, title, copy]) => (
            <li key={step} className="grid gap-1 py-5 sm:grid-cols-[6.5rem_1fr] sm:gap-6">
              <span className="text-xs text-muted-foreground">{step}</span>
              <div>
                <h3 className="text-sm font-medium">{title}</h3>
                <p className="mt-1 text-sm leading-6 text-muted-foreground">{copy}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  )
}
