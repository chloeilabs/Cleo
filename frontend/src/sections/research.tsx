import { SectionHeading } from "@/components/section-heading"
import { compactNumber, fixed, number } from "@/lib/format"
import type { ModelProfile } from "@/types"

export function Research({ profile }: { profile: ModelProfile }) {
  const { identity, metrics, generalization, architecture, training, dataset } = profile

  return (
    <section id="research" className="border-t border-border py-16 sm:py-20">
      <div className="mx-auto max-w-5xl px-5 sm:px-8">
        <SectionHeading
          eyebrow="Research"
          title="Technical note for the alpha release."
          description={
            <p>
              A concise summary of methods and results. Full provenance lives in the model card and
              release manifests.
            </p>
          }
        />

        <article className="mt-10 border-t border-border pt-8">
          <p className="text-xs text-muted-foreground">
            {identity.company_name} · {identity.release}
          </p>
          <h3 className="mt-3 max-w-3xl text-xl font-semibold tracking-tight sm:text-2xl">
            Training a transparent general-language alpha from scratch on a single workstation
          </h3>

          <div className="mt-8 grid gap-10 lg:grid-cols-[1.4fr_.6fr]">
            <div className="space-y-6 text-sm leading-7 text-muted-foreground">
              <section>
                <h4 className="font-medium text-foreground">Abstract</h4>
                <p className="mt-2">
                  {identity.model_name} ({identity.model_id}) is a {number(metrics.parameter_count)}
                  -parameter decoder-only transformer trained from random initialization by{" "}
                  {identity.company_name}. The {identity.release} release expands a TinyStories
                  foundation with WikiText continued pretraining, Dolly answer-only instruction
                  tuning, a {architecture.block_size}-token context, and a gated identity repair.
                </p>
              </section>
              <section>
                <h4 className="font-medium text-foreground">Method</h4>
                <ol className="mt-2 list-decimal space-y-2 pl-5">
                  <li>
                    Train a {architecture.n_layer}-layer transformer and custom{" "}
                    {number(architecture.vocab_size)}-token byte BPE on {dataset.name}.
                  </li>
                  <li>
                    Continue pretraining on {dataset.general.name} while retaining foundation
                    microbatches.
                  </li>
                  <li>
                    Answer-only instruction-tune on {dataset.instruction.name} with retention
                    losses.
                  </li>
                  <li>
                    Promote only when general, instruction, foundation, and identity gates pass.
                  </li>
                </ol>
              </section>
            </div>

            <aside className="space-y-5 border-t border-border pt-6 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-8">
              <p className="text-xs tracking-[0.04em] text-muted-foreground uppercase">Key results</p>
              {[
                [`${fixed(generalization.general_loss_reduction_percent, 1)}% ↓`, "General loss"],
                [
                  `${fixed(generalization.instruction_loss_reduction_percent, 1)}% ↓`,
                  "Instruction loss",
                ],
                ["8 / 8", "Identity match"],
                [training.duration, `${compactNumber(training.tokens_seen)} tokens`],
              ].map(([value, label]) => (
                <div key={label}>
                  <p className="text-xl font-semibold tracking-tight">{value}</p>
                  <p className="mt-0.5 text-xs text-muted-foreground">{label}</p>
                </div>
              ))}
              <a
                href="#model-card"
                className="inline-flex text-sm font-medium underline-offset-4 hover:underline"
              >
                Model card →
              </a>
            </aside>
          </div>
        </article>
      </div>
    </section>
  )
}
