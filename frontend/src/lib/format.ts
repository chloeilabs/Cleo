export function compactNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: value >= 10_000_000 ? 0 : 1,
  }).format(value)
}

export function number(value: number): string {
  return new Intl.NumberFormat("en-US").format(value)
}

export function fixed(value: number, digits = 2): string {
  return value.toFixed(digits)
}

export function excerpt(text: string, maxLength = 470): string {
  if (text.length <= maxLength) return text
  const slice = text.slice(0, maxLength)
  const boundary = slice.lastIndexOf(" ")
  return `${slice.slice(0, boundary > maxLength * 0.75 ? boundary : maxLength).trim()}…`
}
