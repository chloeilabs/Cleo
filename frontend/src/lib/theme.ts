export type ThemePreference = "light" | "dark" | "system"
export type ResolvedTheme = "light" | "dark"

export const THEME_STORAGE_KEY = "cleo-theme"

export function getStoredTheme(): ThemePreference {
  try {
    const value = localStorage.getItem(THEME_STORAGE_KEY)
    if (value === "light" || value === "dark" || value === "system") return value
  } catch {
    // Ignore private-mode / blocked storage.
  }
  return "system"
}

export function resolveTheme(preference: ThemePreference): ResolvedTheme {
  if (preference === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
  }
  return preference
}

export function applyTheme(preference: ThemePreference) {
  const resolved = resolveTheme(preference)
  const root = document.documentElement
  root.classList.toggle("dark", resolved === "dark")
  root.style.colorScheme = resolved
  root.dataset.theme = preference
  const meta = document.querySelector('meta[name="theme-color"]')
  if (meta) meta.setAttribute("content", resolved === "dark" ? "#000000" : "#ffffff")
  return resolved
}

export function storeTheme(preference: ThemePreference) {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, preference)
  } catch {
    // Ignore private-mode / blocked storage.
  }
}

export function cycleTheme(preference: ThemePreference): ThemePreference {
  if (preference === "system") return "light"
  if (preference === "light") return "dark"
  return "system"
}
