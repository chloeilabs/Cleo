import { ThreadView } from "@/components/workspace/thread-view";

export default async function AgentPage({ params }: PageProps<"/agents/[agentId]">) {
  const { agentId } = await params;
  return <ThreadView agentId={decodeURIComponent(agentId)} />;
}
