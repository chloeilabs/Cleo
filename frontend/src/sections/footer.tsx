import { ArrowUpRight } from "lucide-react"

import { Button } from "@/components/ui/button"
import { navigate } from "@/lib/routing"
import type { ModelProfile } from "@/types"

export function Footer({ profile }: { profile: ModelProfile }) {
  return (
    <footer className="border-t border-border bg-background py-14">
      <div className="mx-auto flex max-w-[1120px] flex-col justify-between gap-8 px-5 sm:px-8 md:flex-row md:items-end">
        <div>
          <p className="text-[15px] font-medium">{profile.identity.company_name}</p>
          <p className="mt-2 max-w-md text-sm leading-6 text-muted-foreground">
            {profile.identity.model_name} · {profile.identity.model_id} · Local general-language
            alpha.
          </p>
        </div>
        <Button
          className="h-9 rounded-full px-4 text-[13px]"
          onClick={() => navigate("chat")}
        >
          Try {profile.identity.model_name}
          <ArrowUpRight className="size-3.5" />
        </Button>
      </div>
    </footer>
  )
}
