import { useEffect, useRef, useState } from "react"
import { ArrowUp, Check, Copy, Square } from "lucide-react"

import { AboutPanel, SettingsPanel } from "@/components/chat-panels"
import { ChatSidebar } from "@/components/chat-sidebar"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { streamGeneration } from "@/lib/api"
import { cn } from "@/lib/utils"
import type { ModelProfile } from "@/types"

type Role = "user" | "assistant"

interface Message {
  id: string
  role: Role
  content: string
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-1 py-2" aria-hidden="true">
      <span className="typing-dot size-1.5 rounded-full bg-foreground/45" />
      <span className="typing-dot size-1.5 rounded-full bg-foreground/45" />
      <span className="typing-dot size-1.5 rounded-full bg-foreground/45" />
    </div>
  )
}

export function ChatWorkspace({ profile }: { profile: ModelProfile }) {
  const starters = profile.prompt_starters
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState("")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [aboutOpen, setAboutOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [maxNewTokens, setMaxNewTokens] = useState(160)
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(40)
  const [seed, setSeed] = useState(42)
  const [status, setStatus] = useState("")
  const [error, setError] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const composerRef = useRef<HTMLTextAreaElement | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [messages, isGenerating])

  useEffect(() => {
    if (!status || isGenerating || error) return
    if (!/^done/i.test(status)) return
    const timer = window.setTimeout(() => setStatus(""), 2200)
    return () => window.clearTimeout(timer)
  }, [status, isGenerating, error])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const editing = target?.matches("input, textarea, [contenteditable=true]")
      if (event.key === "/" && !editing && !event.metaKey && !event.ctrlKey) {
        event.preventDefault()
        composerRef.current?.focus()
      }
    }
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [])

  const stop = () => {
    controllerRef.current?.abort()
    controllerRef.current = null
    setIsGenerating(false)
    setStatus("Stopped")
  }

  const resetChat = () => {
    stop()
    setMessages([])
    setDraft("")
    setError("")
    setStatus("")
    composerRef.current?.focus()
  }

  const generate = async (promptText: string) => {
    const prompt = promptText.trim()
    if (!prompt || isGenerating) return

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: prompt,
    }
    const assistantId = crypto.randomUUID()
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", content: "" },
    ])
    setDraft("")
    setError("")
    setStatus("Thinking…")
    setIsGenerating(true)

    const controller = new AbortController()
    controllerRef.current = controller

    try {
      await streamGeneration(
        {
          prompt,
          max_new_tokens: maxNewTokens,
          temperature,
          top_k: topK,
          seed,
        },
        (event) => {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, content: event.text } : message,
            ),
          )
          setStatus(event.status)
        },
        controller.signal,
      )
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return
      const message = caught instanceof Error ? caught.message : "Generation failed."
      setError(message)
      setStatus("Generation failed")
      setMessages((current) =>
        current.filter((item) => !(item.id === assistantId && !item.content)),
      )
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
      setIsGenerating(false)
    }
  }

  const copyMessage = async (message: Message) => {
    if (!message.content) return
    await navigator.clipboard.writeText(message.content)
    setCopiedId(message.id)
    window.setTimeout(
      () => setCopiedId((current) => (current === message.id ? null : current)),
      1400,
    )
  }

  const empty = messages.length === 0
  const showStatus = Boolean(error || (status && (isGenerating || !/^done/i.test(status))))

  return (
    <div className="flex h-svh bg-background text-foreground">
      <ChatSidebar
        profile={profile}
        collapsed={sidebarCollapsed}
        onCollapsedChange={setSidebarCollapsed}
        onNewChat={resetChat}
        onPickStarter={(prompt) => {
          setDraft(prompt)
          composerRef.current?.focus()
        }}
        onOpenAbout={() => setAboutOpen(true)}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <main className="relative flex min-w-0 flex-1 flex-col">
        <header className="flex h-11 items-center justify-center px-14 lg:px-4">
          <p className="text-xs text-muted-foreground">
            {profile.identity.model_name}
            <span className="mx-1.5 text-border">/</span>
            {profile.identity.release}
          </p>
        </header>

        <div className="relative flex-1 overflow-y-auto">
          {empty ? (
            <div className="mx-auto flex min-h-full w-full max-w-xl flex-col justify-center px-4 pb-36 pt-10">
              <h1 className="text-center text-[1.75rem] font-semibold tracking-tight">
                How can I help you today?
              </h1>
              <p className="mx-auto mt-2 max-w-sm text-center text-sm leading-6 text-muted-foreground">
                Local {profile.identity.model_name} on {profile.runtime.device}.
              </p>
              <div className="mx-auto mt-8 grid w-full gap-2 sm:grid-cols-2">
                {starters.slice(0, 4).map((starter) => (
                  <button
                    key={starter.prompt}
                    type="button"
                    onClick={() => void generate(starter.prompt)}
                    className="rounded-xl border border-border px-4 py-3 text-left text-sm leading-6 transition-colors hover:bg-muted"
                  >
                    {starter.label}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto w-full max-w-2xl px-4 py-5 pb-40">
              <div className="space-y-5">
                {messages.map((message) => {
                  const isUser = message.role === "user"
                  const showTyping =
                    !isUser &&
                    isGenerating &&
                    !message.content &&
                    messages.at(-1)?.id === message.id
                  return (
                    <div
                      key={message.id}
                      className={cn(
                        "animate-message-in group flex gap-3",
                        isUser ? "justify-end" : "justify-start",
                      )}
                    >
                      <div className={cn("min-w-0 max-w-[min(100%,40rem)]", isUser && "ml-auto")}>
                        {!isUser && (
                          <p className="mb-1 text-xs font-medium text-muted-foreground">
                            {profile.identity.model_name}
                          </p>
                        )}
                        <div
                          className={cn(
                            "whitespace-pre-wrap text-[15px] leading-7",
                            isUser
                              ? "rounded-2xl bg-muted px-4 py-2.5 text-foreground"
                              : "py-0.5 text-foreground",
                          )}
                        >
                          {showTyping ? <TypingIndicator /> : message.content || "\u00a0"}
                        </div>
                        {!isUser && message.content && (
                          <div className="mt-0.5 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="size-7 rounded-lg text-muted-foreground"
                                  onClick={() => void copyMessage(message)}
                                  aria-label="Copy response"
                                >
                                  {copiedId === message.id ? (
                                    <Check className="size-3.5" />
                                  ) : (
                                    <Copy className="size-3.5" />
                                  )}
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                {copiedId === message.id ? "Copied" : "Copy"}
                              </TooltipContent>
                            </Tooltip>
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-background via-background/95 to-transparent pb-3 pt-14">
          <div className="pointer-events-auto mx-auto w-full max-w-2xl px-4">
            {(showStatus || (status && /^done/i.test(status))) && (
              <div className="mb-2 flex items-center justify-between gap-3 px-1 text-xs text-muted-foreground">
                <span className={cn(error && "text-destructive")} role={error ? "alert" : "status"}>
                  {error || status}
                </span>
                {isGenerating && (
                  <button
                    type="button"
                    onClick={stop}
                    className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 font-medium text-foreground hover:bg-accent"
                  >
                    <Square className="size-2.5 fill-current" />
                    Stop
                  </button>
                )}
              </div>
            )}
            <div className="composer-ring rounded-2xl border border-border bg-background p-1.5">
              <Textarea
                ref={composerRef}
                value={draft}
                maxLength={2_000}
                rows={1}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault()
                    void generate(draft)
                  }
                }}
                placeholder={`Message ${profile.identity.model_name}`}
                className="min-h-[40px] max-h-40 resize-none border-0 bg-transparent px-3 py-2.5 text-[15px] leading-6 shadow-none focus-visible:ring-0"
              />
              <div className="flex items-center justify-end gap-2 px-1 pb-0.5">
                {isGenerating ? (
                  <Button
                    size="icon"
                    className="size-8 rounded-full"
                    onClick={stop}
                    aria-label="Stop generating"
                  >
                    <Square className="size-3 fill-current" />
                  </Button>
                ) : (
                  <Button
                    size="icon"
                    className="size-8 rounded-full disabled:bg-muted disabled:text-muted-foreground"
                    onClick={() => void generate(draft)}
                    disabled={!draft.trim()}
                    aria-label="Send message"
                  >
                    <ArrowUp className="size-4" />
                  </Button>
                )}
              </div>
            </div>
            <p className="mt-2 text-center text-[11px] text-muted-foreground">
              Can make mistakes. Verify important information.
            </p>
          </div>
        </div>
      </main>

      <SettingsPanel
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        maxNewTokens={maxNewTokens}
        temperature={temperature}
        topK={topK}
        seed={seed}
        onMaxNewTokens={setMaxNewTokens}
        onTemperature={setTemperature}
        onTopK={setTopK}
        onSeed={setSeed}
      />
      <AboutPanel open={aboutOpen} onOpenChange={setAboutOpen} profile={profile} />
    </div>
  )
}
