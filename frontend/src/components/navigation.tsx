import { useEffect, useState } from "react"
import { ArrowUpRight, Menu } from "lucide-react"

import { ThemeToggle } from "@/components/theme-toggle"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { navigate } from "@/lib/routing"

const links = [
  ["research", "Research"],
  ["benchmarks", "Benchmarks"],
  ["architecture", "Architecture"],
  ["model-card", "Model card"],
] as const

export function Navigation({ companyName, modelName }: { companyName: string; modelName: string }) {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const update = () => setScrolled(scrollY > 8)
    update()
    addEventListener("scroll", update, { passive: true })
    return () => removeEventListener("scroll", update)
  }, [])

  return (
    <>
      <a
        href="#overview"
        className="fixed left-4 top-3 z-[70] -translate-y-20 rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background transition-transform focus:translate-y-0"
      >
        Skip to content
      </a>
      <header className="fixed inset-x-0 top-0 z-50 bg-background/80 backdrop-blur-xl">
        <nav
          className={`mx-auto grid h-14 max-w-[1120px] grid-cols-[1fr_auto_1fr] items-center gap-4 px-5 sm:px-8 ${
            scrolled ? "border-b border-border/50" : ""
          }`}
        >
          <a href="#top" className="justify-self-start text-[15px] font-medium tracking-[-0.01em]">
            {companyName}
          </a>

          <div className="hidden items-center gap-7 md:flex">
            {links.map(([id, label]) => (
              <a
                key={label}
                href={`#${id}`}
                className="text-[13px] text-foreground/85 transition-colors hover:text-foreground"
              >
                {label}
              </a>
            ))}
          </div>

          <div className="hidden items-center justify-self-end gap-2 md:flex">
            <ThemeToggle />
            <Button
              className="h-8 rounded-full px-3.5 text-[13px] font-medium"
              onClick={() => navigate("chat")}
            >
              Try {modelName}
              <ArrowUpRight className="size-3.5 opacity-80" />
            </Button>
          </div>

          <div className="col-start-3 flex items-center justify-self-end gap-1 md:hidden">
            <ThemeToggle />
            <Sheet>
              <SheetTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-9 rounded-full"
                  aria-label="Open navigation"
                >
                  <Menu className="size-4" />
                </Button>
              </SheetTrigger>
              <SheetContent className="w-[88%] bg-background sm:max-w-sm">
                <SheetHeader className="border-b border-border px-6 py-5 text-left">
                  <SheetTitle className="text-lg">{companyName}</SheetTitle>
                  <SheetDescription>{modelName}</SheetDescription>
                </SheetHeader>
                <div className="flex flex-col p-3">
                  {links.map(([id, label]) => (
                    <SheetClose asChild key={id}>
                      <a
                        href={`#${id}`}
                        className="rounded-lg px-3 py-3 text-base font-medium hover:bg-muted"
                      >
                        {label}
                      </a>
                    </SheetClose>
                  ))}
                  <SheetClose asChild>
                    <Button className="mt-3 h-10 rounded-full" onClick={() => navigate("chat")}>
                      Try {modelName}
                      <ArrowUpRight className="size-3.5" />
                    </Button>
                  </SheetClose>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </nav>
      </header>
    </>
  )
}
