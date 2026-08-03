import { useEffect, useState } from "react"
import { AlertCircle, LoaderCircle } from "lucide-react"

import { Navigation } from "@/components/navigation"
import { Button } from "@/components/ui/button"
import { TooltipProvider } from "@/components/ui/tooltip"
import { getProfile } from "@/lib/api"
import { Architecture } from "@/sections/architecture"
import { Benchmarks } from "@/sections/benchmarks"
import { Footer } from "@/sections/footer"
import { Hero } from "@/sections/hero"
import { ModelCardSection } from "@/sections/model-card"
import { Playground } from "@/sections/playground"
import { Samples } from "@/sections/samples"
import { Training } from "@/sections/training"
import type { ModelProfile } from "@/types"

function App() {
  const [profile, setProfile] = useState<ModelProfile | null>(null)
  const [error, setError] = useState("")

  useEffect(() => {
    const controller = new AbortController()
    getProfile(controller.signal)
      .then((loaded) => {
        setProfile(loaded)
        document.title = `${loaded.identity.company_name} — ${loaded.identity.model_name}`
      })
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return
        setError(caught instanceof Error ? caught.message : "Could not load the model profile.")
      })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!profile || !location.hash) return
    const reveal = () => document.getElementById(location.hash.slice(1))?.scrollIntoView({ block: "start" })
    requestAnimationFrame(reveal)
    const timer = window.setTimeout(reveal, 250)
    return () => window.clearTimeout(timer)
  }, [profile])

  if (error) {
    return (
      <main className="grid min-h-svh place-items-center bg-brand-paper p-6">
        <div className="max-w-md rounded-3xl border bg-card p-8 text-center shadow-xl">
          <AlertCircle className="mx-auto size-9 text-destructive" />
          <h1 className="mt-5 text-2xl font-semibold">Cleo AI could not start</h1>
          <p className="mt-3 text-sm leading-6 text-muted-foreground">{error}</p>
          <Button className="mt-6" onClick={() => location.reload()}>Try again</Button>
        </div>
      </main>
    )
  }

  if (!profile) {
    return (
      <main className="grid min-h-svh place-items-center bg-brand-ink text-white">
        <div className="text-center">
          <div className="mx-auto mb-5 flex size-14 items-center justify-center rounded-2xl bg-brand-lime text-brand-ink">
            <LoaderCircle className="size-6 animate-spin" />
          </div>
          <strong className="text-xl">Loading Cleo 1</strong>
          <p className="mt-2 text-sm text-white/45">Reading the local checkpoint profile…</p>
        </div>
      </main>
    )
  }

  return (
    <TooltipProvider delayDuration={250}>
      <Navigation companyName={profile.identity.company_name} modelName={profile.identity.model_name} />
      <main>
        <Hero profile={profile} />
        <Benchmarks profile={profile} />
        <Architecture profile={profile} />
        <Training profile={profile} />
        <ModelCardSection profile={profile} />
        <Samples profile={profile} />
        <Playground profile={profile} />
      </main>
      <Footer profile={profile} />
    </TooltipProvider>
  )
}

export default App
