import { useEffect, useState } from "react"
import { AlertCircle, LoaderCircle } from "lucide-react"

import { ChatWorkspace } from "@/components/chat-workspace"
import { HomePage } from "@/components/home-page"
import { Button } from "@/components/ui/button"
import { getProfile } from "@/lib/api"
import { type AppView, viewFromHash } from "@/lib/routing"
import type { ModelProfile } from "@/types"

function App() {
  const [profile, setProfile] = useState<ModelProfile | null>(null)
  const [error, setError] = useState("")
  const [view, setView] = useState<AppView>(() =>
    typeof window === "undefined" ? "home" : viewFromHash(),
  )

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
    const onHashChange = () => setView(viewFromHash())
    onHashChange()
    window.addEventListener("hashchange", onHashChange)
    return () => window.removeEventListener("hashchange", onHashChange)
  }, [])

  useEffect(() => {
    document.body.dataset.view = view
    if (view === "home") window.scrollTo(0, 0)
  }, [view])

  if (error) {
    return (
      <main className="grid min-h-svh place-items-center bg-background p-6">
        <div className="max-w-md rounded-3xl border border-border bg-card p-8 text-center">
          <AlertCircle className="mx-auto size-8 text-destructive" />
          <h1 className="mt-5 text-xl font-semibold tracking-tight">Cleo AI could not start</h1>
          <p className="mt-3 text-sm leading-6 text-muted-foreground">{error}</p>
          <Button className="mt-6 rounded-full" onClick={() => location.reload()}>
            Try again
          </Button>
        </div>
      </main>
    )
  }

  if (!profile) {
    return (
      <main className="grid min-h-svh place-items-center bg-background">
        <div className="text-center">
          <div className="mx-auto mb-4 grid size-12 place-items-center rounded-full bg-muted">
            <LoaderCircle className="size-5 animate-spin text-muted-foreground" />
          </div>
          <strong className="text-base font-semibold tracking-tight">Loading Cleo 1</strong>
          <p className="mt-1.5 text-sm text-muted-foreground">Reading the local checkpoint…</p>
        </div>
      </main>
    )
  }

  if (view === "chat") return <ChatWorkspace profile={profile} />
  return <HomePage profile={profile} />
}

export default App
