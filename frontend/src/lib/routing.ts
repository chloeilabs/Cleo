export type AppView = "home" | "chat"

export function viewFromHash(hash = window.location.hash): AppView {
  const normalized = hash.replace(/^#\/?/, "").toLowerCase()
  return normalized === "chat" || normalized.startsWith("chat/") ? "chat" : "home"
}

export function hashForView(view: AppView): string {
  return view === "chat" ? "#/chat" : "#/"
}

export function navigate(view: AppView) {
  const next = hashForView(view)
  if (window.location.hash !== next) {
    window.location.hash = next
  } else {
    window.dispatchEvent(new HashChangeEvent("hashchange"))
  }
}
