import {
  Home,
  Info,
  Menu,
  MessageSquarePlus,
  PanelLeftClose,
  Settings2,
} from "lucide-react"

import { ThemeToggle } from "@/components/theme-toggle"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { navigate } from "@/lib/routing"
import { cn } from "@/lib/utils"
import type { ModelProfile } from "@/types"

interface ChatSidebarProps {
  profile: ModelProfile
  collapsed: boolean
  onCollapsedChange: (collapsed: boolean) => void
  onNewChat: () => void
  onPickStarter: (prompt: string) => void
  onOpenAbout: () => void
  onOpenSettings: () => void
}

function SidebarBody({
  profile,
  onNewChat,
  onPickStarter,
  onOpenAbout,
  onOpenSettings,
  onCollapse,
}: {
  profile: ModelProfile
  onNewChat: () => void
  onPickStarter: (prompt: string) => void
  onOpenAbout: () => void
  onOpenSettings: () => void
  onCollapse?: () => void
}) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-1 px-3 pt-3">
        <Button
          variant="ghost"
          className="h-9 flex-1 justify-start gap-2 rounded-lg px-2.5 text-sm font-medium hover:bg-muted"
          onClick={onNewChat}
        >
          <MessageSquarePlus className="size-4" />
          New chat
        </Button>
        {onCollapse && (
          <Button
            variant="ghost"
            size="icon"
            className="hidden size-9 rounded-lg text-muted-foreground hover:bg-muted lg:inline-flex"
            onClick={onCollapse}
            aria-label="Collapse sidebar"
          >
            <PanelLeftClose className="size-4" />
          </Button>
        )}
      </div>

      <div className="mt-3 px-4">
        <p className="text-sm font-medium tracking-tight">{profile.identity.model_name}</p>
        <p className="text-xs text-muted-foreground">{profile.runtime.device}</p>
      </div>

      <div className="mt-5 flex-1 overflow-y-auto px-2 pb-3">
        <p className="px-2 text-[11px] tracking-[0.04em] text-muted-foreground uppercase">
          Starters
        </p>
        <div className="mt-1 space-y-0.5">
          {profile.prompt_starters.map((starter) => (
            <button
              key={starter.prompt}
              type="button"
              onClick={() => onPickStarter(starter.prompt)}
              className="w-full rounded-lg px-2.5 py-2 text-left text-sm text-foreground/80 transition-colors hover:bg-muted"
            >
              <span className="line-clamp-2">{starter.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-0.5 border-t border-border p-2">
        <ThemeToggle showLabel className="hover:bg-muted" />
        <Button
          variant="ghost"
          className="h-9 w-full justify-start gap-2 rounded-lg px-2.5 text-sm font-medium text-foreground/80 hover:bg-muted"
          onClick={() => navigate("home")}
        >
          <Home className="size-4" />
          Home
        </Button>
        <Button
          variant="ghost"
          className="h-9 w-full justify-start gap-2 rounded-lg px-2.5 text-sm font-medium text-foreground/80 hover:bg-muted"
          onClick={onOpenSettings}
        >
          <Settings2 className="size-4" />
          Sampling
        </Button>
        <Button
          variant="ghost"
          className="h-9 w-full justify-start gap-2 rounded-lg px-2.5 text-sm font-medium text-foreground/80 hover:bg-muted"
          onClick={onOpenAbout}
        >
          <Info className="size-4" />
          Details
        </Button>
      </div>
    </div>
  )
}

export function ChatSidebar(props: ChatSidebarProps) {
  const {
    profile,
    collapsed,
    onCollapsedChange,
    onNewChat,
    onPickStarter,
    onOpenAbout,
    onOpenSettings,
  } = props

  return (
    <>
      <aside
        className={cn(
          "hidden h-svh shrink-0 border-r border-border/80 bg-background transition-[width] duration-200 ease-out lg:block",
          collapsed ? "w-0 overflow-hidden border-r-0" : "w-[260px]",
        )}
      >
        {!collapsed && (
          <SidebarBody
            profile={profile}
            onNewChat={onNewChat}
            onPickStarter={onPickStarter}
            onOpenAbout={onOpenAbout}
            onOpenSettings={onOpenSettings}
            onCollapse={() => onCollapsedChange(true)}
          />
        )}
      </aside>

      <div className="fixed left-3 top-3 z-40 flex items-center gap-2 lg:hidden">
        <Sheet>
          <SheetTrigger asChild>
            <Button
              variant="outline"
              size="icon"
              className="size-9 rounded-lg border-border bg-background"
              aria-label="Open menu"
            >
              <Menu className="size-4" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-[272px] bg-background p-0 sm:max-w-none">
            <SheetHeader className="sr-only">
              <SheetTitle>Menu</SheetTitle>
            </SheetHeader>
            <SidebarBody
              profile={profile}
              onNewChat={onNewChat}
              onPickStarter={onPickStarter}
              onOpenAbout={onOpenAbout}
              onOpenSettings={onOpenSettings}
            />
          </SheetContent>
        </Sheet>
      </div>

      {collapsed && (
        <Button
          variant="outline"
          size="icon"
          className="fixed left-3 top-3 z-40 hidden size-9 rounded-lg border-border bg-background lg:inline-flex"
          onClick={() => onCollapsedChange(false)}
          aria-label="Open sidebar"
        >
          <Menu className="size-4" />
        </Button>
      )}
    </>
  )
}
