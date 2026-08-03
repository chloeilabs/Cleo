import { Monitor, Moon, Sun } from "lucide-react"

import { useTheme } from "@/components/theme-provider"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

const labels = {
  system: "System theme",
  light: "Light theme",
  dark: "Dark theme",
} as const

export function ThemeToggle({
  className,
  showLabel = false,
}: {
  className?: string
  showLabel?: boolean
}) {
  const { preference, cycle } = useTheme()
  const Icon = preference === "dark" ? Moon : preference === "light" ? Sun : Monitor
  const label = labels[preference]

  const button = (
    <Button
      type="button"
      variant={showLabel ? "ghost" : "ghost"}
      size={showLabel ? "default" : "icon"}
      onClick={cycle}
      aria-label={`Theme: ${label}. Click to change.`}
      className={cn(
        showLabel
          ? "h-10 w-full justify-start gap-2 rounded-xl px-3 text-sm font-medium"
          : "size-9 rounded-full text-muted-foreground hover:text-foreground",
        className,
      )}
    >
      <Icon className="size-4" />
      {showLabel && <span>Theme · {preference}</span>}
    </Button>
  )

  if (showLabel) return button

  return (
    <Tooltip>
      <TooltipTrigger asChild>{button}</TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}
