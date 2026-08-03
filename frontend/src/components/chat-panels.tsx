import { Label } from "@/components/ui/label"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { compactNumber, fixed, number } from "@/lib/format"
import type { ModelProfile } from "@/types"

interface SettingsPanelProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  maxNewTokens: number
  temperature: number
  topK: number
  seed: number
  onMaxNewTokens: (value: number) => void
  onTemperature: (value: number) => void
  onTopK: (value: number) => void
  onSeed: (value: number) => void
}

function SettingRow({
  label,
  hint,
  value,
  display,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  hint: string
  value: number
  display: string
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div>
          <Label className="text-sm font-medium">{label}</Label>
          <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>
        </div>
        <span className="rounded-lg bg-muted px-2 py-1 font-mono text-xs tabular-nums">
          {display}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(values) => onChange(values[0])}
        aria-label={label}
      />
    </div>
  )
}

export function SettingsPanel({
  open,
  onOpenChange,
  maxNewTokens,
  temperature,
  topK,
  seed,
  onMaxNewTokens,
  onTemperature,
  onTopK,
  onSeed,
}: SettingsPanelProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full border-l border-border bg-background sm:max-w-md">
        <SheetHeader className="border-b border-border px-1 pb-4 text-left">
          <SheetTitle className="text-lg">Sampling</SheetTitle>
          <SheetDescription>
            Shape the continuation without changing the checkpoint.
          </SheetDescription>
        </SheetHeader>
        <div className="mt-6 space-y-8 px-1">
          <SettingRow
            label="Length"
            hint="Maximum new tokens"
            value={maxNewTokens}
            display={String(maxNewTokens)}
            min={16}
            max={256}
            step={16}
            onChange={onMaxNewTokens}
          />
          <SettingRow
            label="Creativity"
            hint="Higher values are more surprising"
            value={temperature}
            display={temperature.toFixed(2)}
            min={0.3}
            max={1.5}
            step={0.05}
            onChange={onTemperature}
          />
          <SettingRow
            label="Top-k"
            hint="0 considers the full vocabulary"
            value={topK}
            display={String(topK)}
            min={0}
            max={100}
            step={1}
            onChange={onTopK}
          />
          <div className="space-y-2">
            <Label htmlFor="seed">Seed</Label>
            <p className="text-xs text-muted-foreground">Reuse a seed for repeatable output</p>
            <Input
              id="seed"
              type="number"
              min={0}
              max={4_294_967_295}
              value={seed}
              onChange={(event) =>
                onSeed(Math.min(4_294_967_295, Math.max(0, Number(event.target.value))))
              }
              className="h-11 rounded-xl"
            />
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}

export function AboutPanel({
  open,
  onOpenChange,
  profile,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  profile: ModelProfile
}) {
  const rows = [
    ["Parameters", compactNumber(profile.metrics.parameter_count)],
    ["Context", `${profile.architecture.block_size} tokens`],
    ["Vocabulary", `${number(profile.architecture.vocab_size)} BPE`],
    ["Training step", number(profile.metrics.training_step)],
    ["Device", profile.runtime.device],
    ["General val. loss", fixed(profile.generalization.general_validation_loss, 4)],
    ["Instruction val. loss", fixed(profile.generalization.instruction_validation_loss, 4)],
    ["Release", profile.identity.release],
  ]

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full overflow-y-auto border-l border-border bg-background sm:max-w-md">
        <SheetHeader className="border-b border-border px-1 pb-4 text-left">
          <SheetTitle className="text-lg">{profile.identity.model_name}</SheetTitle>
          <SheetDescription>
            Local research alpha from {profile.identity.company_name}. Not a production assistant.
          </SheetDescription>
        </SheetHeader>
        <dl className="mt-6 space-y-3 px-1">
          {rows.map(([label, value]) => (
            <div
              key={label}
              className="flex items-center justify-between gap-4 rounded-xl bg-muted/70 px-3 py-2.5"
            >
              <dt className="text-sm text-muted-foreground">{label}</dt>
              <dd className="text-sm font-medium tabular-nums">{value}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-6 px-1 text-xs leading-5 text-muted-foreground">
          Checkpoint {profile.runtime.checkpoint}. Responses can be repetitive or incorrect at this
          scale.
        </p>
      </SheetContent>
    </Sheet>
  )
}
