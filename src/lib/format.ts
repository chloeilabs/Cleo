/** Presentation helpers shared across client components. */

const RELATIVE_UNITS: Array<[Intl.RelativeTimeFormatUnit, number]> = [
  ["year", 365 * 24 * 60 * 60 * 1000],
  ["month", 30 * 24 * 60 * 60 * 1000],
  ["week", 7 * 24 * 60 * 60 * 1000],
  ["day", 24 * 60 * 60 * 1000],
  ["hour", 60 * 60 * 1000],
  ["minute", 60 * 1000],
];

const relative = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

export function relativeTime(timestamp?: number): string {
  if (!timestamp) return "—";

  const delta = timestamp - Date.now();
  const magnitude = Math.abs(delta);

  if (magnitude < 45_000) return "just now";

  for (const [unit, ms] of RELATIVE_UNITS) {
    if (magnitude >= ms) {
      return relative.format(Math.round(delta / ms), unit);
    }
  }

  return relative.format(Math.round(delta / 1000), "second");
}

export function absoluteTime(timestamp?: number): string | undefined {
  if (!timestamp) return undefined;
  return new Date(timestamp).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function duration(ms?: number): string {
  if (ms === undefined || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;

  const totalSeconds = Math.round(ms / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s`;

  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`;

  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

export function compactNumber(value?: number): string {
  if (value === undefined) return "—";
  return new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function exactNumber(value?: number): string {
  if (value === undefined) return "—";
  return new Intl.NumberFormat("en").format(value);
}

export function cents(value?: number): string {
  if (value === undefined) return "—";
  if (value === 0) return "$0.00";
  const dollars = value / 100;
  return `$${dollars.toFixed(dollars < 0.01 ? 4 : 2)}`;
}

export function bytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB"];
  let size = value / 1024;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size < 10 ? 1 : 0)} ${units[unit]}`;
}

/** `https://github.com/acme/api` → `acme/api` */
export function repoLabel(url: string): string {
  return url
    .trim()
    .replace(/\.git$/, "")
    .replace(/^https?:\/\/(www\.)?(github|gitlab|bitbucket)\.com\//, "")
    .replace(/^git@[^:]+:/, "")
    .replace(/^https?:\/\//, "");
}

export function classNames(
  ...values: Array<string | false | null | undefined>
): string {
  return values.filter(Boolean).join(" ");
}

/** Title for a run: the first line of the user's prompt, trimmed. */
export function firstLine(text: string, limit = 80): string {
  const line = text.split("\n").find((entry) => entry.trim().length > 0) ?? text;
  const trimmed = line.trim();
  return trimmed.length > limit ? `${trimmed.slice(0, limit - 1)}…` : trimmed;
}
