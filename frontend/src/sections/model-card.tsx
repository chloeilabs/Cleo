import { SectionHeading } from "@/components/section-heading"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table"
import { compactNumber, fixed, number } from "@/lib/format"
import type { ModelProfile } from "@/types"

const intendedUse = [
  "Educational experiments with small language models",
  "Local general-language and instruction-following research",
  "Controlled summarization, extraction, classification, and generation probes",
  "Tokenizer, attention, training, and inference study",
  "Local, offline demonstrations on Apple Silicon",
]

export function ModelCardSection({ profile }: { profile: ModelProfile }) {
  const { identity, runtime, metrics, architecture, dataset, adaptation } = profile
  const identityRows = [
    ["Model", identity.model_name],
    ["Model ID", identity.model_id],
    ["Company", identity.company_name],
    ["Release", identity.release],
    [
      "Identity",
      adaptation.identity_tuned
        ? `${number(profile.generalization.identity_repair_steps)}-step verified repair`
        : "Metadata only",
    ],
    ["Held-out match", `${fixed(adaptation.held_out_exact_match * 100, 0)}%`],
    ["Checkpoint", runtime.checkpoint],
    ["Runtime", `PyTorch / ${runtime.device}`],
    ["Parameters", number(metrics.parameter_count)],
  ]
  const datasetRows = [
    [
      "Foundation",
      `${dataset.name} · ${compactNumber(dataset.train_tokens)} tokens`,
    ],
    ["Foundation license", dataset.license],
    [
      "General",
      `${dataset.general.name} · ${compactNumber(dataset.general.train_tokens)} tokens`,
    ],
    ["General license", dataset.general.license],
    [
      "Instructions",
      `${dataset.instruction.name} · ${number(dataset.instruction.train_examples)} examples`,
    ],
    ["Instruction license", dataset.instruction.license],
  ]

  return (
    <section id="model-card" className="border-t border-border py-16 sm:py-20">
      <div className="mx-auto max-w-5xl px-5 sm:px-8">
        <SectionHeading
          eyebrow="Model card"
          title="Read before you run."
          description={
            <p>
              Boundaries should be as legible as capabilities. {identity.model_name} remains a
              research alpha—not a reliable general assistant.
            </p>
          }
        />

        <div className="mt-10 grid gap-10 lg:grid-cols-2">
          <div>
            <h3 className="text-sm font-medium">Identity & release</h3>
            <Table className="mt-3">
              <TableBody>
                {identityRows.map(([label, value]) => (
                  <TableRow key={label} className="border-border">
                    <TableCell className="w-[36%] pl-0 text-muted-foreground">{label}</TableCell>
                    <TableCell className="break-all pr-0 font-medium">{value}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          <div>
            <h3 className="text-sm font-medium">Intended use</h3>
            <ul className="mt-4 space-y-3 border-t border-border pt-4 text-sm leading-6 text-muted-foreground">
              {intendedUse.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-medium">Training data</h3>
            <Table className="mt-3">
              <TableBody>
                {datasetRows.map(([label, value]) => (
                  <TableRow key={label} className="border-border">
                    <TableCell className="w-[36%] pl-0 text-muted-foreground">{label}</TableCell>
                    <TableCell className="break-all pr-0 font-medium">{value}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          <div>
            <h3 className="text-sm font-medium">Limitations</h3>
            <ul className="mt-4 space-y-3 border-t border-border pt-4 text-sm leading-6 text-muted-foreground">
              <li>Not reliable for production chatbots, factual sources, or safety systems.</li>
              <li>Often repeats, answers incorrectly, or fails multi-step instructions.</li>
              <li>Arithmetic, coding, and multilingual behavior are not established.</li>
              <li>Short {architecture.block_size}-token context. No preference training.</li>
              <li>Do not use for consequential medical, legal, or financial decisions.</li>
            </ul>
          </div>
        </div>

        <Accordion type="single" collapsible className="mt-8 border-t border-border">
          <AccordionItem value="evaluation" className="border-0">
            <AccordionTrigger className="px-0 py-5 text-left text-sm font-medium hover:no-underline">
              Evaluation statement
            </AccordionTrigger>
            <AccordionContent className="px-0 pb-6 text-sm leading-7 text-muted-foreground">
              <p>
                Reported {fixed(metrics.best_validation_loss, 4)} cross-entropy and{" "}
                {fixed(metrics.best_validation_perplexity, 4)} perplexity use fixed WikiText
                validation batches. Instruction loss{" "}
                {fixed(profile.generalization.instruction_validation_loss, 4)} is answer-token
                cross-entropy on a deterministic Dolly split.
              </p>
              <p className="mt-3">
                Identity was evaluated on eight held-out paraphrases. Exact self-identification is
                trained behavior, not self-awareness.
              </p>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </section>
  )
}
