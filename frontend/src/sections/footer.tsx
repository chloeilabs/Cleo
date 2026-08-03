import type { ModelProfile } from "@/types"
import { number } from "@/lib/format"

export function Footer({ profile }: { profile: ModelProfile }) {
  return (
    <footer className="bg-brand-ink py-16 text-white">
      <div className="mx-auto flex max-w-[1240px] flex-col justify-between gap-10 px-5 sm:px-8 md:flex-row md:items-end">
        <div>
          <strong className="text-3xl font-semibold tracking-[-.045em]">{profile.identity.company_name}</strong>
          <p className="mt-3 max-w-xl text-sm leading-6 text-white/45">
            {profile.identity.model_name} · <code className="text-white/65">{profile.identity.model_id}</code> · A local research release built to make the complete path from bytes to stories understandable.
          </p>
        </div>
        <div className="font-mono text-[10px] uppercase leading-6 tracking-[.08em] text-white/45 md:text-right">
          {number(profile.metrics.parameter_count)} parameters<br />
          {number(profile.metrics.training_step)} training steps<br />
          {profile.dataset.license}
        </div>
      </div>
    </footer>
  )
}
