import { useEffect, useState } from "react"

import { compactNumber, excerpt, fixed, number } from "@/lib/format"
import { cn } from "@/lib/utils"
import type { ModelProfile } from "@/types"

const toc = [
  ["overview", "Overview"],
  ["capabilities", "Capabilities"],
  ["benchmarks", "Benchmarks"],
  ["research", "Research"],
  ["architecture", "Architecture"],
  ["training", "Training"],
  ["samples", "Probes"],
  ["model-card", "Model card"],
] as const

export function HomeArticle({ profile }: { profile: ModelProfile }) {
  const [active, setActive] = useState("overview")
  const {
    identity,
    metrics,
    generalization,
    architecture,
    training,
    dataset,
    adaptation,
    benchmark,
    samples,
  } = profile
  const storyChange = (generalization.story_retention_ratio - 1) * 100

  useEffect(() => {
    const sections = toc
      .map(([id]) => document.getElementById(id))
      .filter((section): section is HTMLElement => Boolean(section))

    const update = () => {
      let current = "overview"
      for (const section of sections) {
        if (section.getBoundingClientRect().top <= 140) current = section.id
        else break
      }
      setActive(current)
    }
    update()
    addEventListener("scroll", update, { passive: true })
    addEventListener("resize", update, { passive: true })
    return () => {
      removeEventListener("scroll", update)
      removeEventListener("resize", update)
    }
  }, [])

  return (
    <section className="border-t border-border bg-background text-foreground">
      <div className="mx-auto grid max-w-[1120px] gap-10 px-5 py-16 sm:px-8 md:grid-cols-[200px_minmax(0,1fr)] md:gap-14 lg:grid-cols-[220px_minmax(0,1fr)] lg:gap-16 lg:py-20">
        <aside className="hidden md:block">
          <nav className="sticky top-24 space-y-0.5">
            {toc.map(([id, label]) => (
              <a
                key={id}
                href={`#${id}`}
                aria-current={active === id ? "page" : undefined}
                className={cn(
                  "block rounded-md px-2 py-1.5 text-[13px] transition-colors",
                  active === id
                    ? "bg-muted font-medium text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {label}
              </a>
            ))}
          </nav>
        </aside>

        <article className="min-w-0 max-w-[720px] space-y-16 text-[15px] leading-7 text-muted-foreground">
          <section id="overview">
            <h2 className="text-[28px] font-semibold tracking-[-0.03em] text-foreground">
              Overview
            </h2>
            <p className="mt-4">
              {identity.model_name} ({identity.model_id}) is a {number(metrics.parameter_count)}
              -parameter decoder-only transformer developed and trained by {identity.company_name}{" "}
              from random initialization. The {identity.release} release expands a TinyStories
              foundation with WikiText continued pretraining, Dolly answer-only instruction tuning,
              a {architecture.block_size}-token context, and a gated identity repair.
            </p>
            <p className="mt-4">
              No pretrained weights or pretrained tokenizer were used. This is an experimental
              small-model research release, not a reliable general-purpose assistant.
            </p>
          </section>

          <section id="capabilities">
            <h2 className="text-[28px] font-semibold tracking-[-0.03em] text-foreground">
              Capabilities
            </h2>
            <p className="mt-4">
              Gains are real on gated distributions. Qualitative probes still expose the limits of a{" "}
              {number(metrics.parameter_count)}-parameter alpha.
            </p>
            <div className="mt-8 grid gap-8 sm:grid-cols-2">
              <div>
                <h3 className="text-sm font-medium text-foreground">Strengths</h3>
                <ul className="mt-3 space-y-3">
                  {[
                    ["Self-identification", "Held-out paraphrases match after gated repair."],
                    ["Short instructions", "Dolly answer-only tuning lowers instruction loss."],
                    ["Inspectability", "Tokenizer, stages, checksums, and sampling in one repo."],
                    ["Broader language", "WikiText continued pretraining improves general fit."],
                  ].map(([title, detail]) => (
                    <li key={title}>
                      <p className="text-sm font-medium text-foreground">{title}</p>
                      <p className="mt-1 text-sm">{detail}</p>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h3 className="text-sm font-medium text-foreground">Not claimed</h3>
                <ul className="mt-3 space-y-3">
                  {[
                    ["Reliable reasoning", "Multi-step logic and arithmetic are not established."],
                    ["Long-form quality", "Open-ended text often repeats or drifts."],
                    ["Coding & tools", "Programming and agent workflows are out of scope."],
                    ["Safety alignment", "No RLHF or preference training."],
                  ].map(([title, detail]) => (
                    <li key={title}>
                      <p className="text-sm font-medium text-foreground">{title}</p>
                      <p className="mt-1 text-sm">{detail}</p>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </section>

          <section id="benchmarks">
            <h2 className="text-[28px] font-semibold tracking-[-0.03em] text-foreground">
              Benchmarks
            </h2>
            <p className="mt-4">
              Same tokenizer and fixed held-out batches before and after generalization. Internal
              distribution-fit measurements—not a public leaderboard claim.
            </p>
            <div className="mt-8 space-y-6">
              {[
                [
                  "WikiText general-language validation",
                  generalization.general_baseline_loss,
                  generalization.general_validation_loss,
                  generalization.general_loss_reduction_percent,
                  `Perplexity ${fixed(generalization.general_validation_perplexity, 2)}`,
                ],
                [
                  "Dolly instruction validation",
                  generalization.instruction_baseline_loss,
                  generalization.instruction_validation_loss,
                  generalization.instruction_loss_reduction_percent,
                  "Answer-token cross-entropy",
                ],
              ].map(([label, before, after, reduction, note]) => (
                <div key={String(label)} className="rounded-xl border border-border px-5 py-5">
                  <div className="flex items-center justify-between gap-4">
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-foreground">{label}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{note}</p>
                    </div>
                    <p className="shrink-0 text-3xl font-semibold tracking-tight text-foreground tabular-nums">
                      {fixed(Number(reduction), 1)}%
                    </p>
                  </div>
                  <div className="mt-5 flex items-end justify-between gap-6 text-sm">
                    <div>
                      <span className="text-muted-foreground">Before</span>
                      <strong className="mt-1 block font-medium text-foreground tabular-nums">
                        {fixed(Number(before), 4)}
                      </strong>
                    </div>
                    <div className="text-right">
                      <span className="text-muted-foreground">Release</span>
                      <strong className="mt-1 block font-medium text-foreground tabular-nums">
                        {fixed(Number(after), 4)}
                      </strong>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
              {[
                [`${fixed(benchmark.cached_tokens_per_second, 1)} tok/s`, "KV-cached"],
                [`${fixed(benchmark.uncached_tokens_per_second, 1)} tok/s`, "Full-forward"],
                [`${fixed(benchmark.cache_speedup, 2)}×`, "Cache speedup"],
              ].map(([value, label]) => (
                <div
                  key={label}
                  className="flex min-h-[96px] flex-col justify-between rounded-xl border border-border px-4 py-4"
                >
                  <p className="text-xs text-muted-foreground">{label}</p>
                  <p className="text-xl font-semibold tracking-tight text-foreground tabular-nums">
                    {value}
                  </p>
                </div>
              ))}
            </div>
            <p className="mt-6">
              Identity gate: <span className="text-foreground">8 / 8</span> held-out prompts
              matched. Foundation loss is{" "}
              <span className="text-foreground">{fixed(storyChange, 1)}% higher</span> than the
              pre-generalization release—under the 35% ceiling.
            </p>
          </section>

          <section id="research">
            <h2 className="text-[28px] font-semibold tracking-[-0.03em] text-foreground">
              Research
            </h2>
            <p className="mt-4">
              We train a transparent general-language alpha from scratch on a single workstation,
              then promote only when general, instruction, foundation-retention, and identity gates
              pass together.
            </p>
            <ol className="mt-6 list-decimal space-y-3 pl-5">
              <li>
                Train a {architecture.n_layer}-layer transformer and custom{" "}
                {number(architecture.vocab_size)}-token byte BPE on {dataset.name}.
              </li>
              <li>
                Continue pretraining on {dataset.general.name} while retaining foundation
                microbatches.
              </li>
              <li>
                Answer-only instruction-tune on {dataset.instruction.name} with retention losses.
              </li>
              <li>Promote only when every release gate passes.</li>
            </ol>
            <div className="mt-8 grid grid-cols-2 gap-4 sm:grid-cols-4">
              {[
                [`${fixed(generalization.general_loss_reduction_percent, 1)}% ↓`, "General loss"],
                [
                  `${fixed(generalization.instruction_loss_reduction_percent, 1)}% ↓`,
                  "Instruction loss",
                ],
                ["8 / 8", "Identity"],
                [training.duration, `${compactNumber(training.tokens_seen)} tokens`],
              ].map(([value, label]) => (
                <div key={label}>
                  <p className="text-xl font-semibold tracking-tight text-foreground">{value}</p>
                  <p className="mt-1 text-xs">{label}</p>
                </div>
              ))}
            </div>
          </section>

          <section id="architecture">
            <h2 className="text-[28px] font-semibold tracking-[-0.03em] text-foreground">
              Architecture
            </h2>
            <p className="mt-4">
              A compact decoder-only transformer in PyTorch. Tokenization, attention, training, and
              sampling all live here—pretrained components never enter the pipeline.
            </p>
            <dl className="mt-8 grid grid-cols-2 gap-x-8 gap-y-4 sm:grid-cols-4">
              {[
                ["Parameters", number(metrics.parameter_count)],
                ["Context", `${architecture.block_size} tokens`],
                ["Layers", String(architecture.n_layer)],
                ["Width", String(architecture.n_embd)],
                ["Heads", String(architecture.n_head)],
                ["FFN", number(architecture.ffn_size)],
                ["Vocab", number(architecture.vocab_size)],
                ["Precision", "FP32"],
              ].map(([label, value]) => (
                <div key={label}>
                  <dt className="text-xs">{label}</dt>
                  <dd className="mt-1 text-sm font-medium text-foreground">{value}</dd>
                </div>
              ))}
            </dl>
          </section>

          <section id="training">
            <h2 className="text-[28px] font-semibold tracking-[-0.03em] text-foreground">
              Training
            </h2>
            <p className="mt-4">
              Random initialization to the promoted alpha in {training.duration}.{" "}
              {number(training.tokens_seen)} token presentations with explicit retention gates.
            </p>
            <ol className="mt-6 space-y-5">
              {[
                [
                  "Foundation",
                  `${number(generalization.foundation_steps)} steps from random weights on ${number(dataset.train_stories)} TinyStories examples.`,
                ],
                ["Tokenizer", "Byte-level BPE learned locally from an 8 MiB corpus sample."],
                [
                  "Generalize",
                  `${number(generalization.continued_pretraining_steps)} continued-pretraining steps on ${number(dataset.general.train_documents)} WikiText documents.`,
                ],
                [
                  "Instruct",
                  `${number(generalization.instruction_tuning_steps)} instruction-tuning steps on ${number(dataset.instruction.train_examples)} Dolly examples.`,
                ],
                ...(generalization.accepted && adaptation.identity_tuned
                  ? ([
                      [
                        "Repair",
                        `${number(generalization.identity_repair_steps)}-step identity repair with ${fixed(adaptation.held_out_exact_match * 100, 0)}% held-out exact match.`,
                      ],
                    ] as const)
                  : []),
              ].map(([title, copy]) => (
                <li key={title}>
                  <p className="text-sm font-medium text-foreground">{title}</p>
                  <p className="mt-1 text-sm">{copy}</p>
                </li>
              ))}
            </ol>
          </section>

          <section id="samples">
            <h2 className="text-[28px] font-semibold tracking-[-0.03em] text-foreground">
              Probes
            </h2>
            <p className="mt-4">
              Deterministic, unedited responses from fixed prompts and seeds—including repetition
              and incorrect answers.
            </p>
            <div className="mt-6 space-y-6">
              {(samples.length ? samples.slice(0, 3) : []).map((sample) => (
                <div key={`${sample.seed}-${sample.prompt}`}>
                  <p className="text-xs">Seed {sample.seed}</p>
                  <p className="mt-1 text-sm font-medium text-foreground">{sample.prompt}</p>
                  <p className="mt-2 text-sm">{excerpt(sample.text, 280)}</p>
                </div>
              ))}
            </div>
          </section>

          <section id="model-card">
            <h2 className="text-[28px] font-semibold tracking-[-0.03em] text-foreground">
              Model card
            </h2>
            <p className="mt-4">
              Boundaries should be as legible as capabilities. {identity.model_name} remains a
              research alpha—not a reliable general assistant.
            </p>
            <dl className="mt-6 space-y-3 text-sm">
              {[
                ["Model", identity.model_name],
                ["Model ID", identity.model_id],
                ["Company", identity.company_name],
                ["Release", identity.release],
                ["Checkpoint", profile.runtime.checkpoint],
                ["Device", profile.runtime.device],
                ["Parameters", number(metrics.parameter_count)],
                ["Foundation", `${dataset.name} · ${dataset.license}`],
                ["General", `${dataset.general.name} · ${dataset.general.license}`],
                ["Instructions", `${dataset.instruction.name} · ${dataset.instruction.license}`],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="grid grid-cols-[8rem_1fr] gap-3 border-b border-border/50 py-2.5"
                >
                  <dt>{label}</dt>
                  <dd className="break-all font-medium text-foreground">{value}</dd>
                </div>
              ))}
            </dl>
            <ul className="mt-6 space-y-2 text-sm">
              <li>Not reliable for production chatbots, factual sources, or safety systems.</li>
              <li>Often repeats, answers incorrectly, or fails multi-step instructions.</li>
              <li>No dedicated safety alignment or human-preference training.</li>
              <li>Do not use for consequential medical, legal, or financial decisions.</li>
            </ul>
          </section>
        </article>
      </div>
    </section>
  )
}
