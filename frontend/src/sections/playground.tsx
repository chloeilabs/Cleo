import { useEffect, useRef, useState } from "react"
import {
  Check,
  Copy,
  Dices,
  Eraser,
  Octagon,
  Play,
  Sparkles,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Textarea } from "@/components/ui/textarea"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { streamStory } from "@/lib/api"
import type { ModelProfile } from "@/types"

function randomUint32(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0]
}

interface ControlProps {
  label: string
  hint: string
  value: number
  display: string
  min: number
  max: number
  step: number
  onValueChange: (value: number) => void
}

function SamplingControl({
  label,
  hint,
  value,
  display,
  min,
  max,
  step,
  onValueChange,
}: ControlProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div>
          <Label className="text-sm">{label}</Label>
          <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>
        </div>
        <span className="min-w-14 rounded-lg border bg-muted/55 px-2 py-1 text-center font-mono text-xs">
          {display}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(values) => onValueChange(values[0])}
        aria-label={label}
        className="**:data-[slot=slider-range]:bg-brand-green"
      />
      <div className="flex justify-between font-mono text-[10px] text-muted-foreground/65">
        <span>{min}</span><span>{max}</span>
      </div>
    </div>
  )
}

export function Playground({ profile }: { profile: ModelProfile }) {
  const starters = profile.prompt_starters
  const initialPrompt = starters[0]?.prompt ?? "Once upon a time"
  const [prompt, setPrompt] = useState(initialPrompt)
  const [starter, setStarter] = useState(initialPrompt)
  const [maxNewTokens, setMaxNewTokens] = useState(300)
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(40)
  const [seed, setSeed] = useState(42)
  const [output, setOutput] = useState("")
  const [status, setStatus] = useState("Ready · checkpoint loaded locally")
  const [error, setError] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [copied, setCopied] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)
  const promptRef = useRef<HTMLTextAreaElement | null>(null)
  const outputRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const focusPrompt = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const editing = target?.matches("input, textarea, [contenteditable=true]")
      if (event.key === "/" && !editing && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault()
        promptRef.current?.focus()
        document.getElementById("playground")?.scrollIntoView({ behavior: "smooth" })
      }
    }
    document.addEventListener("keydown", focusPrompt)
    return () => document.removeEventListener("keydown", focusPrompt)
  }, [])

  const stop = () => {
    controllerRef.current?.abort()
    controllerRef.current = null
    if (isGenerating) setStatus("Stopped · partial continuation kept")
    setIsGenerating(false)
  }

  const generate = async () => {
    if (!prompt.trim() || isGenerating) return
    const controller = new AbortController()
    controllerRef.current = controller
    setError("")
    setOutput(prompt)
    setStatus("Connecting to Cleo 1…")
    setIsGenerating(true)
    outputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" })

    try {
      await streamStory(
        {
          prompt,
          max_new_tokens: maxNewTokens,
          temperature,
          top_k: topK,
          seed,
        },
        (event) => {
          setOutput(event.text)
          setStatus(event.status)
        },
        controller.signal,
      )
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return
      const message = caught instanceof Error ? caught.message : "Generation failed."
      setError(message)
      setStatus("Generation failed · check the message below")
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
      setIsGenerating(false)
    }
  }

  const chooseStarter = (value: string) => {
    stop()
    setStarter(value)
    setPrompt(value)
    setOutput("")
    setError("")
    setStatus("Starter loaded · edit it freely, then generate")
  }

  const surprise = () => {
    stop()
    const selected = starters[randomUint32() % starters.length]
    const nextSeed = randomUint32()
    setStarter(selected.prompt)
    setPrompt(selected.prompt)
    setSeed(nextSeed)
    setOutput("")
    setError("")
    setStatus(`Surprise starter loaded · seed ${nextSeed.toLocaleString()}`)
  }

  const clear = () => {
    stop()
    setStarter("")
    setPrompt("")
    setOutput("")
    setError("")
    setStatus("Ready · choose a starter or write your own beginning")
  }

  const copyOutput = async () => {
    if (!output) return
    await navigator.clipboard.writeText(output)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1_600)
  }

  return (
    <section id="playground" className="scroll-mt-20 py-28 sm:py-32">
      <div className="mx-auto max-w-[1240px] px-5 sm:px-8">
        <div className="mb-12">
          <div className="mb-5 flex items-center gap-2.5 text-[11px] font-bold uppercase tracking-[.18em] text-brand-green">
            <span className="size-1.5 rounded-full bg-current" /> Live playground
          </div>
          <h2 className="text-6xl font-semibold leading-[.92] tracking-[-.06em] sm:text-7xl lg:text-8xl">Give it a beginning.</h2>
          <p className="mt-6 max-w-2xl text-base leading-7 text-muted-foreground">The checkpoint stays in local memory and streams tokens from the {profile.runtime.device} device. Reuse the same seed and settings to reproduce a result.</p>
        </div>

        <div className="grid items-start gap-5 lg:grid-cols-[1.15fr_.85fr]">
          <div className="space-y-5">
            <Card className="rounded-3xl bg-card shadow-none">
              <CardContent className="space-y-5 pt-6">
                <div className="grid items-end gap-3 sm:grid-cols-[1fr_auto]">
                  <div className="space-y-2">
                    <Label>Prompt starter</Label>
                    <p className="text-xs text-muted-foreground">Pick a tested opening or write your own</p>
                    <Select value={starter || undefined} onValueChange={chooseStarter}>
                      <SelectTrigger className="h-11 w-full bg-brand-paper"><SelectValue placeholder="Choose a story opening" /></SelectTrigger>
                      <SelectContent>
                        {starters.map((item) => <SelectItem key={item.prompt} value={item.prompt}>{item.label}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <Button variant="secondary" className="h-11 px-4" onClick={surprise}>
                    <Dices /> Surprise me
                  </Button>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="story-prompt">Story beginning</Label>
                  <p className="text-xs text-muted-foreground">Press Enter to generate · Shift+Enter for a new line · / to focus</p>
                  <Textarea
                    ref={promptRef}
                    id="story-prompt"
                    value={prompt}
                    maxLength={2_000}
                    onChange={(event) => setPrompt(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault()
                        void generate()
                      }
                    }}
                    placeholder="Once upon a time…"
                    className="min-h-40 resize-y bg-brand-paper text-base leading-7"
                  />
                </div>
                <p className="text-xs leading-5 text-muted-foreground">Short, concrete openings leave more of the {profile.architecture.block_size}-token context for the story.</p>
              </CardContent>
            </Card>

            <Card ref={outputRef} className="rounded-3xl bg-card shadow-none">
              <CardHeader className="flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-base">Model continuation</CardTitle>
                  <p className="mt-1 text-xs text-muted-foreground">Streaming directly from {profile.identity.model_name}</p>
                </div>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" onClick={() => void copyOutput()} disabled={!output} aria-label="Copy story">
                      {copied ? <Check className="text-brand-green" /> : <Copy />}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{copied ? "Copied" : "Copy story"}</TooltipContent>
                </Tooltip>
              </CardHeader>
              <CardContent>
                <div
                  id="story-output"
                  className={`min-h-[360px] whitespace-pre-wrap rounded-2xl border bg-brand-paper p-5 font-serif text-[17px] leading-8 ${output ? "text-foreground" : "text-muted-foreground"}`}
                  aria-live="polite"
                >
                  {output || "Your story will appear here token by token…"}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="rounded-3xl bg-card shadow-none lg:sticky lg:top-24">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-xl"><Sparkles className="size-5 text-brand-green" /> Sampling controls</CardTitle>
              <p className="text-sm leading-6 text-muted-foreground">Shape the continuation without changing the checkpoint.</p>
            </CardHeader>
            <CardContent className="space-y-7">
              <SamplingControl label="Length" hint="Maximum new tokens" value={maxNewTokens} display={String(maxNewTokens)} min={32} max={512} step={16} onValueChange={setMaxNewTokens} />
              <SamplingControl label="Creativity" hint="Higher values are more surprising" value={temperature} display={temperature.toFixed(2)} min={0.3} max={1.5} step={0.05} onValueChange={setTemperature} />
              <SamplingControl label="Top-k" hint="0 considers the full vocabulary" value={topK} display={String(topK)} min={0} max={100} step={1} onValueChange={setTopK} />
              <div className="space-y-2">
                <Label htmlFor="seed">Seed</Label>
                <p className="text-xs text-muted-foreground">Reuse a seed for repeatable output</p>
                <Input id="seed" type="number" min={0} max={4_294_967_295} value={seed} onChange={(event) => setSeed(Math.min(4_294_967_295, Math.max(0, Number(event.target.value))))} className="h-11 bg-brand-paper" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Button className="col-span-2 h-11 bg-brand-ink text-white hover:bg-brand-green" onClick={() => void generate()} disabled={isGenerating || !prompt.trim()}>
                  {isGenerating ? <><Sparkles className="animate-pulse" /> Writing…</> : <><Play /> Generate story</>}
                </Button>
                <Button variant="destructive" className="h-10" onClick={stop} disabled={!isGenerating}><Octagon /> Stop</Button>
                <Button variant="secondary" className="h-10" onClick={clear}><Eraser /> Clear</Button>
              </div>
              <div id="generation-status" role="status" aria-live="polite" aria-atomic="true" className="rounded-2xl bg-muted p-4 text-sm leading-6 text-muted-foreground">{status}</div>
              {error && <div role="alert" className="rounded-2xl border border-destructive/30 bg-destructive/8 p-4 text-sm text-destructive">{error}</div>}
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
