import type { ReactNode } from "react";

import { SignIn } from "@/components/auth/sign-in";
import { WorkspaceShell } from "@/components/workspace/shell";
import { currentSession } from "@/lib/server/session";

export const dynamic = "force-dynamic";

export default async function WorkspaceLayout({
  children,
}: {
  children: ReactNode;
}) {
  const session = await currentSession();

  if (!session.authenticated) {
    return (
      <div className="h-full overflow-y-auto">
        <SignIn />
      </div>
    );
  }

  return <WorkspaceShell session={session}>{children}</WorkspaceShell>;
}
