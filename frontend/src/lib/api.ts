import type {
  GenerationEvent,
  GenerationRequest,
  ModelProfile,
} from "@/types"

async function responseError(response: Response): Promise<Error> {
  try {
    const payload = (await response.json()) as {
      detail?: string | Array<{ msg?: string }>
    }
    if (typeof payload.detail === "string") return new Error(payload.detail)
    if (Array.isArray(payload.detail)) {
      const message = payload.detail.map((item) => item.msg).filter(Boolean).join(" · ")
      if (message) return new Error(message)
    }
  } catch {
    // Fall through to the status-based message.
  }
  return new Error(`Request failed with status ${response.status}`)
}

export async function getProfile(signal?: AbortSignal): Promise<ModelProfile> {
  const response = await fetch("/api/profile", { signal })
  if (!response.ok) throw await responseError(response)
  return (await response.json()) as ModelProfile
}

export async function streamStory(
  request: GenerationRequest,
  onEvent: (event: GenerationEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok) throw await responseError(response)
  if (!response.body) throw new Error("The browser did not provide a response stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  const consumeLine = (line: string) => {
    const trimmed = line.trim()
    if (trimmed) onEvent(JSON.parse(trimmed) as GenerationEvent)
  }

  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""
    lines.forEach(consumeLine)
    if (done) break
  }
  consumeLine(buffer)
}
