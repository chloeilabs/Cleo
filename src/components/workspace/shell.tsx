"use client";

import { useRouter } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";

import { SidebarIcon } from "@/components/icons";
import { Button } from "@/components/ui/button";
import { ToastProvider } from "@/components/ui/toast";
import { WorkspaceProvider } from "@/components/workspace/provider";
import { Sidebar } from "@/components/workspace/sidebar";
import { classNames } from "@/lib/format";
import type { SessionState } from "@/lib/types";

export function WorkspaceShell({
  session,
  children,
}: {
  session: SessionState;
  children: ReactNode;
}) {
  const router = useRouter();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        router.push("/");
      }
      if (
        (event.metaKey || event.ctrlKey) &&
        event.key.toLowerCase() === "b"
      ) {
        event.preventDefault();
        setCollapsed((value) => !value);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [router]);

  return (
    <ToastProvider>
      <WorkspaceProvider session={session}>
        <div className="flex h-full">
          <aside
            className={classNames(
              "hidden shrink-0 border-r border-hairline transition-[width] duration-200 md:block",
              collapsed ? "w-0 overflow-hidden" : "w-72",
            )}
          >
            <Sidebar />
          </aside>

          {mobileOpen ? (
            <div className="fixed inset-0 z-50 md:hidden">
              <button
                type="button"
                aria-label="Close navigation"
                onClick={() => setMobileOpen(false)}
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              />
              <aside className="animate-fade-up absolute inset-y-0 left-0 w-72 border-r border-hairline bg-canvas">
                <Sidebar
                  onNavigate={() => setMobileOpen(false)}
                  onClose={() => setMobileOpen(false)}
                />
              </aside>
            </div>
          ) : null}

          <main className="relative flex min-w-0 flex-1 flex-col">
            <div className="absolute top-3 left-3 z-30 flex gap-1.5">
              <Button
                variant="ghost"
                size="icon"
                aria-label="Toggle navigation"
                className="md:hidden"
                onClick={() => setMobileOpen(true)}
              >
                <SidebarIcon className="size-4" />
              </Button>

              {/* Wrapped rather than given `hidden` directly: the button's own
                  `inline-flex` is an unconditional utility and would win. */}
              {collapsed ? (
                <div className="hidden md:block">
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label="Show sidebar"
                    onClick={() => setCollapsed(false)}
                  >
                    <SidebarIcon className="size-4" />
                  </Button>
                </div>
              ) : null}
            </div>

            {children}
          </main>
        </div>
      </WorkspaceProvider>
    </ToastProvider>
  );
}
