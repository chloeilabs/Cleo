import { AlertTriangle, BookMarked, Database, ShieldCheck } from "lucide-react"

import { SectionHeading } from "@/components/section-heading"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table"
import { compactNumber, fixed, number } from "@/lib/format"
import type { ModelProfile } from "@/types"

const intendedUse = [
  "Educational experiments with small language models",
  "Short, child-level fictional story continuation",
  "Tokenizer, attention, training, and inference study",
  "Local, offline demonstrations on Apple Silicon",
]

export function ModelCardSection({ profile }: { profile: ModelProfile }) {
  const { identity, runtime, metrics, architecture, dataset, adaptation } = profile
  const identityRows = [
    ["Model", identity.model_name],
    ["Model ID", identity.model_id],
    ["Company", identity.company_name],
    ["Identity behavior", adaptation.identity_tuned ? `${number(adaptation.completed_steps)}-step verified fine-tune` : "Metadata only"],
    ["Identity evaluation", `${fixed(adaptation.held_out_exact_match * 100, 0)}% held-out exact match`],
    ["Checkpoint", runtime.checkpoint],
    ["Runtime", `PyTorch / ${runtime.device}`],
    ["Parameters", number(metrics.parameter_count)],
  ]
  const datasetRows = [
    ["Dataset", dataset.name],
    ["Train corpus", `${number(dataset.train_stories)} stories / ${compactNumber(dataset.train_tokens)} tokens`],
    ["Validation", `${number(dataset.validation_stories)} stories / ${compactNumber(dataset.validation_tokens)} tokens`],
    ["License", dataset.license],
    ["Revision", dataset.revision],
  ]

  return (
    <section id="model-card" className="scroll-mt-20 bg-[#efede6] py-28 sm:py-32">
      <div className="mx-auto max-w-[1240px] px-5 sm:px-8">
        <SectionHeading
          eyebrow="Model card"
          title="Read before you run."
          description={<p>A model launch should make boundaries as legible as capabilities. {identity.model_name} is a narrow educational generator, not a small general assistant.</p>}
        />
        <div className="mt-14 grid gap-5 lg:grid-cols-2">
          <Card className="rounded-3xl bg-card shadow-none">
            <CardHeader>
              <Badge className="mb-3 w-fit rounded-full bg-brand-lime/55 text-brand-ink">From-scratch checkpoint</Badge>
              <CardTitle className="flex items-center gap-2 text-xl"><BookMarked className="size-5 text-brand-green" /> Identity & release</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableBody>
                  {identityRows.map(([label, value]) => (
                    <TableRow key={label}>
                      <TableCell className="w-[38%] text-muted-foreground">{label}</TableCell>
                      <TableCell className="break-all font-medium">{value}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card className="rounded-3xl bg-card shadow-none">
            <CardHeader><CardTitle className="flex items-center gap-2 text-xl"><ShieldCheck className="size-5 text-brand-green" /> Intended use</CardTitle></CardHeader>
            <CardContent>
              <ul className="space-y-4 text-sm leading-6 text-muted-foreground">
                {intendedUse.map((item) => <li key={item} className="flex gap-3"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-brand-green" />{item}</li>)}
              </ul>
            </CardContent>
          </Card>

          <Card className="rounded-3xl bg-card shadow-none">
            <CardHeader><CardTitle className="flex items-center gap-2 text-xl"><Database className="size-5 text-brand-green" /> Training data & license</CardTitle></CardHeader>
            <CardContent>
              <Table>
                <TableBody>
                  {datasetRows.map(([label, value]) => (
                    <TableRow key={label}>
                      <TableCell className="w-[38%] text-muted-foreground">{label}</TableCell>
                      <TableCell className="break-all font-medium">{value}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card className="rounded-3xl border-brand-coral/35 bg-[#fff8f3] shadow-none">
            <CardHeader><CardTitle className="flex items-center gap-2 text-xl"><AlertTriangle className="size-5 text-brand-coral" /> Limitations & out-of-scope use</CardTitle></CardHeader>
            <CardContent>
              <ul className="space-y-3 text-sm leading-6 text-muted-foreground">
                <li>Not a chatbot, factual source, reasoning engine, or safety system.</li>
                <li>May repeat, contradict itself, lose characters, or produce unsuitable text.</li>
                <li>Short {architecture.block_size}-token attention window.</li>
                <li>No dedicated safety alignment or human-preference training.</li>
                <li>Do not use for consequential medical, legal, financial, or assessment decisions.</li>
              </ul>
            </CardContent>
          </Card>
        </div>
        <Accordion type="single" collapsible className="mt-5 overflow-hidden rounded-3xl border bg-card px-6">
          <AccordionItem value="evaluation" className="border-0">
            <AccordionTrigger className="py-6 text-left text-xl hover:no-underline">
              <span>Evaluation statement <small className="ml-3 hidden text-xs font-normal text-muted-foreground sm:inline">Methodology & provenance</small></span>
            </AccordionTrigger>
            <AccordionContent className="border-t pb-7 pt-5 text-sm leading-7 text-muted-foreground">
              <p>The reported {fixed(metrics.best_validation_loss, 4)} cross-entropy and {fixed(metrics.best_validation_perplexity, 4)} perplexity come from 50 fixed batches of the pinned TinyStories validation token stream. They measure next-token fit inside this narrow distribution. They do not establish broad language understanding and should not be compared directly with differently tokenized models.</p>
              <p className="mt-4">Identity behavior was evaluated separately on held-out paraphrases. Exact self-identification is a narrowly fine-tuned behavior backed by canonical checkpoint metadata; it does not make the model a general assistant or imply self-awareness.</p>
              <div className="mt-5 rounded-xl bg-muted p-4 font-mono text-[11px] leading-5">TOKENIZER SHA-256 AND SOURCE CHECKSUMS ARE RECORDED IN THE DATA MANIFEST · CHECKPOINT AND TOKENIZER ARE VERIFIED TOGETHER AT LOAD TIME</div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </section>
  )
}
