import { SectionHeading } from "@/components/section-heading"
import { excerpt } from "@/lib/format"
import type { ModelProfile } from "@/types"

export function Samples({ profile }: { profile: ModelProfile }) {
  const samples = profile.samples.slice(0, 3)
  return (
    <section id="samples" className="border-t border-border py-16 sm:py-20">
      <div className="mx-auto max-w-5xl px-5 sm:px-8">
        <SectionHeading
          eyebrow="Probes"
          title="What alpha means in practice."
          description={
            <p>
              Deterministic, unedited responses from fixed prompts and seeds—including repetition
              and incorrect answers.
            </p>
          }
        />
        <div className="mt-10 divide-y divide-border border-y border-border">
          {(samples.length
            ? samples
            : [
                {
                  prompt: "Ask Cleo 1 a focused question",
                  seed: 0,
                  text: "Open chat to generate a live continuation.",
                },
              ]
          ).map((sample) => (
            <article key={`${sample.seed}-${sample.prompt}`} className="grid gap-3 py-6 lg:grid-cols-[1fr_1.2fr]">
              <div>
                <p className="text-xs text-muted-foreground">
                  {sample.seed ? `Seed ${sample.seed}` : "Live probe"}
                </p>
                <h3 className="mt-2 text-sm font-medium leading-6">{sample.prompt}</h3>
              </div>
              <p className="text-sm leading-7 text-muted-foreground">{excerpt(sample.text, 360)}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
