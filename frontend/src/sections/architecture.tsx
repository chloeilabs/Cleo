import { SectionHeading } from "@/components/section-heading"
import { number } from "@/lib/format"
import type { ModelProfile } from "@/types"

export function Architecture({ profile }: { profile: ModelProfile }) {
  const { architecture, metrics } = profile
  const stages = [
    ["01", "UTF-8 input", "Lossless text bytes with no unknown token."],
    ["02", `${number(architecture.vocab_size)} BPE tokens`, "256 base bytes, learned merges, BOS and EOS."],
    ["03", `${architecture.n_embd}-wide state`, "Tied token embeddings plus learned positions."],
    [
      "04",
      `${architecture.n_layer} pre-norm blocks`,
      `${architecture.n_head}-head causal attention and ${number(architecture.ffn_size)}-wide GELU MLPs.`,
    ],
    ["05", "Next byte-piece", "Temperature and top-k sampling with a KV cache."],
  ] as const
  const specs = [
    ["Parameters", number(metrics.parameter_count)],
    ["Context", `${architecture.block_size} tokens`],
    ["Attention", `${architecture.n_head} × ${architecture.n_embd / architecture.n_head}`],
    ["FFN width", number(architecture.ffn_size)],
    ["Normalization", "Pre-LayerNorm"],
    ["Activation", "GELU"],
    ["Dropout", architecture.dropout.toFixed(1)],
    ["Precision", "FP32"],
  ]

  return (
    <section id="architecture" className="border-t border-border py-16 sm:py-20">
      <div className="mx-auto max-w-5xl px-5 sm:px-8">
        <SectionHeading
          eyebrow="Architecture"
          title="Every weight has a provenance."
          description={
            <p>
              A compact decoder-only transformer in PyTorch. Tokenization, attention, training, and
              sampling all live here—no pretrained components.
            </p>
          }
        />
        <ol className="mt-10 divide-y divide-border border-y border-border">
          {stages.map(([step, title, copy]) => (
            <li key={step} className="grid gap-1 py-5 sm:grid-cols-[3rem_1fr] sm:gap-6">
              <span className="font-mono text-xs text-muted-foreground">{step}</span>
              <div>
                <h3 className="text-sm font-medium">{title}</h3>
                <p className="mt-1 text-sm leading-6 text-muted-foreground">{copy}</p>
              </div>
            </li>
          ))}
        </ol>
        <dl className="mt-8 grid grid-cols-2 gap-x-8 sm:grid-cols-4">
          {specs.map(([label, value]) => (
            <div key={label} className="border-b border-border py-4">
              <dt className="text-xs text-muted-foreground">{label}</dt>
              <dd className="mt-1 text-sm font-medium">{value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  )
}
