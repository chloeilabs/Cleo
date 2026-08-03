import { HomeArticle } from "@/components/home-article"
import { Navigation } from "@/components/navigation"
import { Footer } from "@/sections/footer"
import { Hero } from "@/sections/hero"
import type { ModelProfile } from "@/types"

export function HomePage({ profile }: { profile: ModelProfile }) {
  return (
    <div className="min-h-svh bg-background text-foreground">
      <Navigation
        companyName={profile.identity.company_name}
        modelName={profile.identity.model_name}
      />
      <main>
        <Hero profile={profile} />
        <HomeArticle profile={profile} />
      </main>
      <Footer profile={profile} />
    </div>
  )
}
