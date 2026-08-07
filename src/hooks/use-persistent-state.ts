"use client";

import { useCallback, useSyncExternalStore } from "react";

/**
 * `localStorage` treated as an external store.
 *
 * `useSyncExternalStore` gives the server (and the first client render) the
 * default value, then swaps in the stored one during hydration without an
 * extra render pass. Snapshots are memoised per key because the hook requires
 * a referentially stable value.
 */

const listeners = new Set<() => void>();
const snapshots = new Map<string, { raw: string | null; value: unknown }>();

function subscribe(onStoreChange: () => void): () => void {
  listeners.add(onStoreChange);
  window.addEventListener("storage", onStoreChange);

  return () => {
    listeners.delete(onStoreChange);
    window.removeEventListener("storage", onStoreChange);
  };
}

function readSnapshot<T>(key: string, fallback: T): T {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(key);
  } catch {
    return fallback;
  }

  const cached = snapshots.get(key);
  if (cached && cached.raw === raw) return cached.value as T;

  let value = fallback;
  if (raw !== null) {
    try {
      value = JSON.parse(raw) as T;
    } catch {
      value = fallback;
    }
  }

  snapshots.set(key, { raw, value });
  return value;
}

export function usePersistentState<T>(
  key: string,
  initial: T,
): [T, (value: T) => void] {
  const value = useSyncExternalStore(
    subscribe,
    () => readSnapshot(key, initial),
    () => initial,
  );

  const setValue = useCallback(
    (next: T) => {
      const raw = JSON.stringify(next);
      snapshots.set(key, { raw, value: next });

      try {
        window.localStorage.setItem(key, raw);
      } catch {
        // Persistence is a convenience, never a requirement.
      }

      for (const listener of listeners) listener();
    },
    [key],
  );

  return [value, setValue];
}
