import { Microscope } from "lucide-react"

import { SectionHeading } from "@/components/section-heading"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { excerpt } from "@/lib/format"
import type { ModelProfile } from "@/types"

export function Samples({ profile }: { profile: ModelProfile }) {
  const samples = profile.samples.slice(0, 3)
  return (
    <section id="samples" className="bg-[#e7e3da] py-28 sm:py-32">
      <div className="mx-auto max-w-[1240px] px-5 sm:px-8">
        <SectionHeading
          eyebrow="Fixed capability probes"
          title="What alpha means in practice."
          description={<p>Deterministic, unedited responses from fixed prompts and seeds. These samples expose repetition and incorrect answers alongside the quantitative loss gains; transparency is part of the release.</p>}
        />
        <div className="mt-14 grid gap-5 lg:grid-cols-3">
          {(samples.length ? samples : [{ prompt: "Ask Cleo 1 a focused question", seed: 0, text: "The live playground is ready for a prompt." }]).map((sample) => (
            <Card key={`${sample.seed}-${sample.prompt}`} className="min-h-[350px] rounded-3xl bg-card shadow-none">
              <CardHeader>
                <span className="font-mono text-[10px] font-bold uppercase tracking-[.12em] text-brand-green">{sample.seed ? `Seed ${sample.seed}` : "Samples unavailable"}</span>
                <h3 className="mt-4 text-xl font-semibold leading-6 tracking-[-.03em]">{sample.prompt}</h3>
              </CardHeader>
              <CardContent className="flex flex-1 flex-col">
                <p className="font-serif text-[15px] leading-7 text-muted-foreground">{excerpt(sample.text)}</p>
                <Microscope className="mt-auto size-6 translate-y-2 text-brand-coral" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
