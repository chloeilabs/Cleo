import { Gauge, LineChart } from "lucide-react"

import { SectionHeading } from "@/components/section-heading"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { fixed } from "@/lib/format"
import type { ModelProfile } from "@/types"

function LossChart({ points }: { points: ModelProfile["validation_curve"] }) {
  const width = 760
  const height = 300
  const left = 48
  const right = 20
  const top = 18
  const bottom = 38
  const maxStep = Math.max(...points.map((point) => point.step), 1)
  const maxLoss = Math.max(7, ...points.map((point) => point.loss))
  const minLoss = Math.min(1, ...points.map((point) => point.loss))
  const xy = (step: number, loss: number) => ({
    x: left + ((width - left - right) * step) / maxStep,
    y: top + ((height - top - bottom) * (maxLoss - loss)) / (maxLoss - minLoss),
  })
  const coordinates = points.map((point) => xy(point.step, point.loss))
  const polyline = coordinates.map(({ x, y }) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ")
  const area = `${left},${height - bottom} ${polyline} ${width - right},${height - bottom}`

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="mt-6 w-full" role="img" aria-label="Validation loss falls throughout training">
      {[7, 5, 3, 1].map((loss) => {
        const y = xy(0, loss).y
        return (
          <g key={loss}>
            <line x1={left} y1={y} x2={width - right} y2={y} stroke="currentColor" className="text-border" />
            <text x={left - 10} y={y + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
              {loss}
            </text>
          </g>
        )
      })}
      <polygon points={area} fill="url(#loss-fill)" />
      <polyline points={polyline} fill="none" stroke="#ff6b4a" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
      {coordinates.map(({ x, y }, index) => (
        <circle key={points[index].step} cx={x} cy={y} r="5" fill="#fffdf7" stroke="#ff6b4a" strokeWidth="3" />
      ))}
      <defs>
        <linearGradient id="loss-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ff6b4a" stopOpacity=".22" />
          <stop offset="1" stopColor="#ff6b4a" stopOpacity="0" />
        </linearGradient>
      </defs>
      <text x={left} y={height - 8} className="fill-muted-foreground text-[11px]">0</text>
      <text x={width - right} y={height - 8} textAnchor="end" className="fill-muted-foreground text-[11px]">
        {maxStep.toLocaleString()} steps
      </text>
    </svg>
  )
}

export function Benchmarks({ profile }: { profile: ModelProfile }) {
  const { metrics, benchmark, validation_curve } = profile
  const baseline = benchmark.cached_tokens_per_second
    ? (100 * benchmark.uncached_tokens_per_second) / benchmark.cached_tokens_per_second
    : 0

  return (
    <section id="benchmarks" className="scroll-mt-20 py-28 sm:py-32">
      <div className="mx-auto max-w-[1240px] px-5 sm:px-8">
        <SectionHeading
          eyebrow="Evaluation"
          title="Measured, not imagined."
          description={
            <p>
              Held-out TinyStories loss tracks what the model learned. A local generation benchmark shows what the inference cache changed. These are reproducible internal measurements—not claims against general-purpose models.
            </p>
          }
        />
        <div className="mt-14 grid gap-5 lg:grid-cols-2">
          <Card className="rounded-3xl bg-card py-0 shadow-none">
            <CardHeader className="p-7 pb-0">
              <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[.14em] text-muted-foreground">
                <LineChart className="size-4 text-brand-coral" /> Validation cross-entropy
              </div>
              <h3 className="mt-3 text-2xl font-semibold tracking-[-.035em]">
                {fixed(metrics.loss_reduction_percent, 1)}% lower loss over training
              </h3>
              <p className="text-xs text-muted-foreground">50 fixed validation batches every 1,000 steps · seed 1337</p>
            </CardHeader>
            <CardContent className="p-5 pt-0 sm:p-7 sm:pt-0">
              <LossChart points={validation_curve} />
            </CardContent>
          </Card>

          <Card className="rounded-3xl bg-card py-0 shadow-none">
            <CardHeader className="p-7 pb-0">
              <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[.14em] text-muted-foreground">
                <Gauge className="size-4 text-brand-green" /> Autoregressive inference
              </div>
              <h3 className="mt-3 text-2xl font-semibold tracking-[-.035em]">Same model. Less repeated work.</h3>
            </CardHeader>
            <CardContent className="p-7">
              <div className="mb-9 mt-2">
                <strong className="text-7xl font-semibold tracking-[-.07em]">{fixed(benchmark.cache_speedup, 2)}×</strong>
                <span className="ml-4 inline-block max-w-36 text-sm leading-5 text-muted-foreground">faster with per-layer KV caching</span>
              </div>
              <div className="space-y-6">
                <div>
                  <div className="mb-2 flex justify-between text-sm"><span>KV cached</span><strong>{fixed(benchmark.cached_tokens_per_second, 1)} tok/s</strong></div>
                  <Progress value={100} className="h-2 bg-muted **:data-[slot=progress-indicator]:bg-brand-green" />
                </div>
                <div>
                  <div className="mb-2 flex justify-between text-sm"><span>Full forward</span><strong>{fixed(benchmark.uncached_tokens_per_second, 1)} tok/s</strong></div>
                  <Progress value={baseline} className="h-2 bg-muted **:data-[slot=progress-indicator]:bg-muted-foreground" />
                </div>
              </div>
              <p className="mt-9 text-xs leading-5 text-muted-foreground">
                {benchmark.device} · FP32 · batch 1 · {benchmark.new_tokens} new tokens · {benchmark.outputs_equal ? "token-identical output" : "outputs not compared"}
              </p>
            </CardContent>
          </Card>
        </div>
        <div className="mt-5 flex flex-col gap-2 rounded-2xl border bg-white/35 p-5 text-sm leading-6 text-muted-foreground sm:flex-row sm:gap-5">
          <strong className="shrink-0 text-foreground">How to read this</strong>
          <span>Loss and perplexity apply only to the pinned TinyStories validation stream. Throughput is one warmed-up local MPS run and varies with prompt length, thermals, and system load.</span>
        </div>
      </div>
    </section>
  )
}
