import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react"

import {
  applyTheme,
  cycleTheme,
  getStoredTheme,
  storeTheme,
  type ResolvedTheme,
  type ThemePreference,
} from "@/lib/theme"

interface ThemeContextValue {
  preference: ThemePreference
  resolved: ResolvedTheme
  setPreference: (preference: ThemePreference) => void
  cycle: () => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [preference, setPreferenceState] = useState<ThemePreference>(() =>
    typeof window === "undefined" ? "system" : getStoredTheme(),
  )
  const [resolved, setResolved] = useState<ResolvedTheme>(() =>
    typeof window === "undefined" ? "light" : applyTheme(getStoredTheme()),
  )

  const setPreference = useCallback((next: ThemePreference) => {
    storeTheme(next)
    setPreferenceState(next)
    setResolved(applyTheme(next))
  }, [])

  const cycle = useCallback(() => {
    setPreference(cycleTheme(preference))
  }, [preference, setPreference])

  useEffect(() => {
    setResolved(applyTheme(preference))
    if (preference !== "system") return
    const media = window.matchMedia("(prefers-color-scheme: dark)")
    const onChange = () => setResolved(applyTheme("system"))
    media.addEventListener("change", onChange)
    return () => media.removeEventListener("change", onChange)
  }, [preference])

  const value = useMemo(
    () => ({ preference, resolved, setPreference, cycle }),
    [preference, resolved, setPreference, cycle],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) throw new Error("useTheme must be used within ThemeProvider")
  return context
}
