import { Binary, Braces, BrainCircuit, Sparkles, WholeWord } from "lucide-react"

import { SectionHeading } from "@/components/section-heading"
import { number } from "@/lib/format"
import type { ModelProfile } from "@/types"

export function Architecture({ profile }: { profile: ModelProfile }) {
  const { architecture, metrics } = profile
  const stages = [
    { icon: Binary, number: "01 / BYTES", title: "UTF-8 input", copy: "Lossless text bytes with no unknown token." },
    { icon: WholeWord, number: "02 / BPE", title: `${number(architecture.vocab_size)} tokens`, copy: "256 base bytes, 766 learned merges, BOS and EOS." },
    { icon: Braces, number: "03 / EMBED", title: `${architecture.n_embd}-wide state`, copy: "Tied token embeddings plus learned positions." },
    { icon: BrainCircuit, number: "04 / REASON", title: `${architecture.n_layer} pre-norm blocks`, copy: `${architecture.n_head}-head causal attention and ${number(architecture.ffn_size)}-wide GELU MLPs.` },
    { icon: Sparkles, number: "05 / SAMPLE", title: "Next byte-piece", copy: "Temperature and top-k sampling with a KV cache." },
  ]
  const specs = [
    ["Parameters", number(metrics.parameter_count)],
    ["Context", `${architecture.block_size} tokens`],
    ["Attention", `${architecture.n_head} × ${architecture.n_embd / architecture.n_head} heads`],
    ["FFN width", number(architecture.ffn_size)],
    ["Normalization", "Pre-LayerNorm"],
    ["Activation", "GELU (tanh)"],
    ["Dropout", architecture.dropout.toFixed(1)],
    ["Precision", "FP32"],
  ]

  return (
    <section id="architecture" className="bg-brand-ink py-28 text-white sm:py-32">
      <div className="mx-auto max-w-[1240px] px-5 sm:px-8">
        <SectionHeading
          inverted
          eyebrow="Architecture"
          title="Every weight has a provenance."
          description={
            <p>A compact decoder-only transformer built directly in PyTorch. Tokenization, causal attention, training, checkpointing, and sampling all live in this repository; pretrained components never enter the pipeline.</p>
          }
        />
        <div className="mt-16 grid gap-px overflow-hidden rounded-3xl border border-white/10 bg-white/10 md:grid-cols-2 lg:grid-cols-5">
          {stages.map(({ icon: Icon, number: stage, title, copy }) => (
            <article key={stage} className="relative min-h-60 bg-brand-ink p-6 transition-colors hover:bg-white/[.045]">
              <Icon className="mb-9 size-5 text-brand-lime" />
              <span className="font-mono text-[9px] font-bold tracking-[.12em] text-brand-lime">{stage}</span>
              <h3 className="mt-3 text-xl font-semibold tracking-[-.03em]">{title}</h3>
              <p className="mt-3 text-sm leading-6 text-white/48">{copy}</p>
            </article>
          ))}
        </div>
        <dl className="mt-10 grid grid-cols-2 gap-x-8 gap-y-0 sm:grid-cols-4">
          {specs.map(([label, value]) => (
            <div key={label} className="border-b border-white/10 py-5">
              <dt className="text-[10px] uppercase tracking-[.12em] text-white/35">{label}</dt>
              <dd className="mt-2 text-sm font-medium text-white/85">{value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  )
}
