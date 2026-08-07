"use client";

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { usePersistentState } from "@/hooks/use-persistent-state";
import { api } from "@/lib/api";
import type {
  AgentSummary,
  ConversationMode,
  ModelOption,
  RepositoryOption,
  SessionState,
} from "@/lib/types";

/** Composer settings that should survive a reload. */
export interface ComposerPreferences {
  modelKey?: string;
  repoUrl?: string;
  startingRef: string;
  mode: ConversationMode;
  autoCreatePR: boolean;
  workOnCurrentBranch: boolean;
}

const DEFAULT_PREFERENCES: ComposerPreferences = {
  startingRef: "",
  mode: "agent",
  autoCreatePR: false,
  workOnCurrentBranch: false,
};

interface WorkspaceValue {
  session: SessionState;
  agents: AgentSummary[];
  agentsLoading: boolean;
  models: ModelOption[];
  repositories: RepositoryOption[];
  catalogLoading: boolean;
  showArchived: boolean;
  setShowArchived: (value: boolean) => void;
  preferences: ComposerPreferences;
  setPreferences: (value: ComposerPreferences) => void;
  selectedModel?: ModelOption;
  refreshAgents: () => Promise<void>;
  /** Merge a known-good agent into the list without waiting for a poll. */
  upsertAgent: (agent: AgentSummary) => void;
  removeAgent: (agentId: string) => void;
}

const WorkspaceContext = createContext<WorkspaceValue | null>(null);

const ACTIVE_POLL_MS = 4_000;
const IDLE_POLL_MS = 20_000;

export function WorkspaceProvider({
  session,
  children,
}: {
  session: SessionState;
  children: ReactNode;
}) {
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [loadedOnce, setLoadedOnce] = useState(false);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [repositories, setRepositories] = useState<RepositoryOption[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [showArchived, setShowArchived] = usePersistentState(
    "cleo.showArchived",
    false,
  );
  const [preferences, setPreferences] = usePersistentState(
    "cleo.composer",
    DEFAULT_PREFERENCES,
  );

  const inFlight = useRef<{
    key: string;
    promise: Promise<AgentSummary[] | undefined>;
  } | null>(null);

  /**
   * Fetch the list, collapsing concurrent calls onto one request.
   *
   * Callers share the pending promise rather than being turned away, so a
   * caller that arrives mid-flight still receives the result. Turning them away
   * would drop the very first load under StrictMode's double-invoked effects.
   */
  const fetchAgents = useCallback(() => {
    const key = String(showArchived);
    if (inFlight.current?.key === key) return inFlight.current.promise;

    const promise = (async () => {
      try {
        return (await api.agents({ includeArchived: showArchived })).items;
      } catch {
        // Keep the last known list on screen; the next poll retries.
        return undefined;
      } finally {
        if (inFlight.current?.key === key) inFlight.current = null;
      }
    })();

    inFlight.current = { key, promise };
    return promise;
  }, [showArchived]);

  const refreshAgents = useCallback(async () => {
    const items = await fetchAgents();
    if (items) setAgents(items);
    setLoadedOnce(true);
  }, [fetchAgents]);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      const items = await fetchAgents();
      if (cancelled) return;
      if (items) setAgents(items);
      setLoadedOnce(true);
    })();

    return () => {
      cancelled = true;
    };
  }, [fetchAgents]);

  // Poll faster while something is running so the sidebar tracks live work.
  const hasRunning = agents.some((agent) => agent.status === "running");

  useEffect(() => {
    const interval = setInterval(
      () => void refreshAgents(),
      hasRunning ? ACTIVE_POLL_MS : IDLE_POLL_MS,
    );
    return () => clearInterval(interval);
  }, [hasRunning, refreshAgents]);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      const [modelResult, repoResult] = await Promise.allSettled([
        api.models(),
        api.repositories(),
      ]);

      if (cancelled) return;

      if (modelResult.status === "fulfilled") {
        setModels(modelResult.value.models);
      }
      if (repoResult.status === "fulfilled") {
        setRepositories(repoResult.value.repositories);
      }
      setCatalogLoading(false);
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const upsertAgent = useCallback((agent: AgentSummary) => {
    setAgents((current) => {
      const index = current.findIndex(
        (entry) => entry.agentId === agent.agentId,
      );
      if (index === -1) return [agent, ...current];
      const next = [...current];
      next[index] = { ...next[index], ...agent };
      return next;
    });
  }, []);

  const removeAgent = useCallback((agentId: string) => {
    setAgents((current) => current.filter((entry) => entry.agentId !== agentId));
  }, []);

  const selectedModel = useMemo(() => {
    if (models.length === 0) return undefined;
    return (
      models.find((model) => model.key === preferences.modelKey) ??
      models.find((model) => model.isDefault) ??
      models[0]
    );
  }, [models, preferences.modelKey]);

  const value = useMemo<WorkspaceValue>(
    () => ({
      session,
      agents,
      agentsLoading: !loadedOnce,
      models,
      repositories,
      catalogLoading,
      showArchived,
      setShowArchived,
      preferences,
      setPreferences,
      selectedModel,
      refreshAgents,
      upsertAgent,
      removeAgent,
    }),
    [
      session,
      agents,
      loadedOnce,
      models,
      repositories,
      catalogLoading,
      showArchived,
      setShowArchived,
      preferences,
      setPreferences,
      selectedModel,
      refreshAgents,
      upsertAgent,
      removeAgent,
    ],
  );

  return (
    <WorkspaceContext.Provider value={value}>
      {children}
    </WorkspaceContext.Provider>
  );
}

export function useWorkspace(): WorkspaceValue {
  const context = useContext(WorkspaceContext);
  if (!context) {
    throw new Error("useWorkspace must be used inside a WorkspaceProvider.");
  }
  return context;
}

/** Build the SDK `ModelSelection` for the currently chosen dropdown entry. */
export function modelSelection(model?: ModelOption) {
  if (!model) return undefined;
  return { id: model.modelId, ...(model.params ? { params: model.params } : {}) };
}
