import { useEffect, useState } from "react"
import { ArrowUp, Menu } from "lucide-react"

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

const links = [
  ["benchmarks", "Benchmarks"],
  ["architecture", "Architecture"],
  ["training", "Training"],
  ["model-card", "Model card"],
] as const

export function Navigation({ companyName, modelName }: { companyName: string; modelName: string }) {
  const [active, setActive] = useState("top")
  const [progress, setProgress] = useState(0)
  const [backToTop, setBackToTop] = useState(false)

  useEffect(() => {
    const sections = [
      "benchmarks",
      "architecture",
      "training",
      "model-card",
      "samples",
      "playground",
    ]
      .map((id) => document.getElementById(id))
      .filter((section): section is HTMLElement => Boolean(section))

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0]
        if (visible) setActive(visible.target.id)
      },
      { rootMargin: "-18% 0px -62% 0px", threshold: [0, 0.15, 0.35, 0.6] },
    )
    sections.forEach((section) => observer.observe(section))

    const update = () => {
      const scrollable = Math.max(document.documentElement.scrollHeight - innerHeight, 1)
      setProgress(Math.min(Math.max(scrollY / scrollable, 0), 1))
      setBackToTop(scrollY > innerHeight * 0.7)
    }
    update()
    addEventListener("scroll", update, { passive: true })
    addEventListener("resize", update, { passive: true })
    return () => {
      observer.disconnect()
      removeEventListener("scroll", update)
      removeEventListener("resize", update)
    }
  }, [])

  return (
    <>
      <a
        href="#benchmarks"
        className="fixed left-4 top-3 z-[70] -translate-y-20 rounded-lg bg-brand-lime px-4 py-2 text-sm font-semibold text-brand-ink transition-transform focus:translate-y-0"
      >
        Skip to model details
      </a>
      <header className="fixed inset-x-0 top-0 z-50 border-b border-white/10 bg-brand-ink/88 text-white backdrop-blur-xl">
        <nav className="mx-auto flex h-[72px] max-w-[1320px] items-center justify-between px-5 sm:px-8">
          <a href="#top" className="flex items-center gap-3 text-lg font-semibold tracking-tight">
            <span className="size-3 rounded-full bg-brand-lime shadow-[0_0_0_5px_rgba(202,255,69,.13)]" />
            {companyName}
          </a>
          <div className="hidden items-center gap-7 lg:flex">
            {links.map(([id, label]) => (
              <a
                key={id}
                href={`#${id}`}
                aria-current={active === id ? "page" : undefined}
                className="relative text-[13px] font-medium text-white/65 transition-colors hover:text-white aria-[current=page]:text-white after:absolute after:-bottom-3 after:inset-x-0 after:h-0.5 after:origin-left after:scale-x-0 after:rounded-full after:bg-brand-lime after:transition-transform aria-[current=page]:after:scale-x-100"
              >
                {label}
              </a>
            ))}
            <Button asChild className="h-9 rounded-full bg-brand-lime px-4 text-brand-ink hover:bg-brand-lime/85">
              <a href="#playground">Try {modelName}</a>
            </Button>
          </div>
          <Sheet>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="text-white hover:bg-white/10 hover:text-white lg:hidden"
                aria-label="Open navigation"
              >
                <Menu />
              </Button>
            </SheetTrigger>
            <SheetContent className="w-[88%] bg-brand-paper sm:max-w-sm">
              <SheetHeader className="border-b px-6 py-6 text-left">
                <SheetTitle className="text-2xl">{companyName}</SheetTitle>
                <SheetDescription>{modelName} model release</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col p-4">
                {[...links, ["samples", "Sample stories"], ["playground", `Try ${modelName}`]].map(
                  ([id, label]) => (
                    <SheetClose asChild key={id}>
                      <a
                        href={`#${id}`}
                        className="rounded-xl px-4 py-4 text-lg font-medium hover:bg-muted"
                      >
                        {label}
                      </a>
                    </SheetClose>
                  ),
                )}
              </div>
            </SheetContent>
          </Sheet>
        </nav>
        <span
          className="absolute inset-x-0 bottom-0 h-0.5 origin-left bg-brand-lime"
          style={{ transform: `scaleX(${progress})` }}
          aria-hidden="true"
        />
      </header>
      <a
        href="#top"
        aria-label="Back to top"
        className={`fixed bottom-5 right-5 z-40 grid size-11 place-items-center rounded-full bg-brand-ink text-white shadow-xl transition-all hover:bg-brand-green focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-brand-coral/30 ${
          backToTop ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-4 opacity-0"
        }`}
      >
        <ArrowUp className="size-4" />
      </a>
    </>
  )
}
