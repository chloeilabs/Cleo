import { SectionHeading } from "@/components/section-heading"
import { Button } from "@/components/ui/button"
import { navigate } from "@/lib/routing"
import type { ModelProfile } from "@/types"

const strengths = [
  ["Self-identification", "Canonical identity answers match held-out paraphrases after gated repair."],
  ["Short instruction following", "Answer-only Dolly tuning lowers instruction validation loss."],
  ["Local inspectability", "Tokenizer, training, checksums, and sampling live in one repo."],
  ["Broader language fit", "WikiText continued pretraining reduces general-language validation loss."],
] as const

const gaps = [
  ["Reliable reasoning", "Multi-step logic, arithmetic, and factual calibration are not established."],
  ["Long-form quality", "Open-ended generations often repeat, drift, or contradict themselves."],
  ["Coding & tools", "Programming, tool use, and agent workflows are out of scope."],
  ["Safety alignment", "No RLHF / preference training—do not use for consequential decisions."],
] as const

export function Capabilities({ profile }: { profile: ModelProfile }) {
  return (
    <section id="capabilities" className="border-t border-border py-16 sm:py-20">
      <div className="mx-auto max-w-5xl px-5 sm:px-8">
        <SectionHeading
          eyebrow="Capabilities"
          title="What this alpha can—and cannot—do."
          description={
            <p>
              Gains are real on gated distributions; probes still expose the limits of a{" "}
              {profile.metrics.parameter_count.toLocaleString()}-parameter research model.
            </p>
          }
        />

        <div className="mt-10 grid gap-10 lg:grid-cols-2 lg:gap-16">
          <div>
            <h3 className="text-sm font-medium">Strengths</h3>
            <ul className="mt-4 space-y-4 border-t border-border pt-4">
              {strengths.map(([title, detail]) => (
                <li key={title}>
                  <p className="text-sm font-medium">{title}</p>
                  <p className="mt-1 text-sm leading-6 text-muted-foreground">{detail}</p>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-medium">Not claimed</h3>
            <ul className="mt-4 space-y-4 border-t border-border pt-4">
              {gaps.map(([title, detail]) => (
                <li key={title}>
                  <p className="text-sm font-medium">{title}</p>
                  <p className="mt-1 text-sm leading-6 text-muted-foreground">{detail}</p>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-10 flex items-center justify-between gap-3 border-t border-border pt-6">
          <p className="text-sm text-muted-foreground">Open the local chat playground.</p>
          <Button className="h-9 rounded-full px-4" onClick={() => navigate("chat")}>
            Open chat
          </Button>
        </div>
      </div>
    </section>
  )
}
