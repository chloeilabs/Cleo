/**
 * A deliberately small syntax highlighter.
 *
 * Agents write mostly TypeScript, Python, shell, and JSON, and a full grammar
 * engine would cost more bundle than the fidelity is worth here. This lexes the
 * handful of token classes that carry almost all of the visual signal — strings,
 * comments, numbers, keywords, and call sites — and leaves everything else
 * plain.
 */

export type TokenKind =
  | "plain"
  | "comment"
  | "string"
  | "number"
  | "keyword"
  | "literal"
  | "function"
  | "punctuation"
  | "tag"
  | "property";

export interface Token {
  kind: TokenKind;
  text: string;
}

const KEYWORDS = new Set([
  // JavaScript / TypeScript
  "abstract", "as", "async", "await", "break", "case", "catch", "class",
  "const", "continue", "declare", "default", "delete", "do", "else", "enum",
  "export", "extends", "finally", "for", "from", "function", "get", "if",
  "implements", "import", "in", "instanceof", "interface", "keyof", "let",
  "namespace", "new", "of", "private", "protected", "public", "readonly",
  "return", "satisfies", "set", "static", "super", "switch", "this", "throw",
  "try", "type", "typeof", "var", "void", "while", "yield",
  // Python
  "def", "elif", "except", "global", "lambda", "nonlocal", "pass", "raise",
  "with", "assert", "del",
  // Go / Rust / others
  "chan", "defer", "fn", "func", "go", "impl", "let mut", "match", "mod",
  "mut", "package", "pub", "struct", "trait", "unsafe", "use", "where",
  // Shell
  "then", "fi", "esac", "done", "elifn", "local", "echo",
]);

const LITERALS = new Set([
  "true", "false", "null", "undefined", "None", "True", "False", "NaN",
  "Infinity", "nil", "self",
]);

const HASH_COMMENT_LANGUAGES = new Set([
  "bash", "sh", "shell", "zsh", "python", "py", "ruby", "rb", "yaml", "yml",
  "toml", "ini", "dockerfile", "make", "makefile", "r", "perl", "conf",
]);

const MARKUP_LANGUAGES = new Set(["html", "xml", "svg", "vue", "svelte"]);

/**
 * One master regex, ordered so that greedier constructs (comments, strings)
 * win before their contents can be mistaken for other tokens.
 */
function buildPattern(language: string): RegExp {
  const parts = [
    String.raw`(?<block>\/\*[\s\S]*?(?:\*\/|$))`,
    String.raw`(?<line>\/\/[^\n]*)`,
    ...(HASH_COMMENT_LANGUAGES.has(language)
      ? [String.raw`(?<hash>#[^\n]*)`]
      : []),
    String.raw`(?<template>\`(?:\\.|[^\\\`])*\`?)`,
    String.raw`(?<double>"(?:\\.|[^\\"\n])*"?)`,
    String.raw`(?<single>'(?:\\.|[^\\'\n])*'?)`,
    String.raw`(?<number>\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b|\b0x[0-9a-fA-F]+\b)`,
    String.raw`(?<word>[A-Za-z_$][\w$]*)`,
    String.raw`(?<punct>[{}()[\].,;:=+\-*/%<>!?&|^~]+)`,
  ];

  return new RegExp(parts.join("|"), "g");
}

export function tokenize(source: string, language = ""): Token[] {
  const normalized = language.toLowerCase();

  if (normalized === "json") return tokenizeJson(source);
  if (MARKUP_LANGUAGES.has(normalized)) return tokenizeMarkup(source);

  const pattern = buildPattern(normalized);
  const tokens: Token[] = [];
  let cursor = 0;

  for (const match of source.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > cursor) {
      tokens.push({ kind: "plain", text: source.slice(cursor, index) });
    }

    const groups = match.groups ?? {};
    const text = match[0];

    if (groups.block || groups.line || groups.hash) {
      tokens.push({ kind: "comment", text });
    } else if (groups.template || groups.double || groups.single) {
      tokens.push({ kind: "string", text });
    } else if (groups.number) {
      tokens.push({ kind: "number", text });
    } else if (groups.word) {
      const followedByCall = source[index + text.length] === "(";
      if (LITERALS.has(text)) {
        tokens.push({ kind: "literal", text });
      } else if (KEYWORDS.has(text)) {
        tokens.push({ kind: "keyword", text });
      } else if (followedByCall) {
        tokens.push({ kind: "function", text });
      } else {
        tokens.push({ kind: "plain", text });
      }
    } else {
      tokens.push({ kind: "punctuation", text });
    }

    cursor = index + text.length;
  }

  if (cursor < source.length) {
    tokens.push({ kind: "plain", text: source.slice(cursor) });
  }

  return tokens;
}

function tokenizeJson(source: string): Token[] {
  const pattern =
    /("(?:\\.|[^\\"])*")(\s*:)?|(\b-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(\btrue\b|\bfalse\b|\bnull\b)|([{}[\],:])/g;

  const tokens: Token[] = [];
  let cursor = 0;

  for (const match of source.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > cursor) {
      tokens.push({ kind: "plain", text: source.slice(cursor, index) });
    }

    const [text, quoted, colon, number, literal, punct] = match;

    if (quoted) {
      tokens.push({ kind: colon ? "property" : "string", text: quoted });
      if (colon) tokens.push({ kind: "punctuation", text: colon });
    } else if (number) {
      tokens.push({ kind: "number", text: number });
    } else if (literal) {
      tokens.push({ kind: "literal", text: literal });
    } else if (punct) {
      tokens.push({ kind: "punctuation", text: punct });
    } else {
      tokens.push({ kind: "plain", text });
    }

    cursor = index + text.length;
  }

  if (cursor < source.length) {
    tokens.push({ kind: "plain", text: source.slice(cursor) });
  }

  return tokens;
}

function tokenizeMarkup(source: string): Token[] {
  const pattern = /(<!--[\s\S]*?-->)|(<\/?[A-Za-z][\w:-]*)|("[^"]*"|'[^']*')|(\/?>)/g;
  const tokens: Token[] = [];
  let cursor = 0;

  for (const match of source.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > cursor) {
      tokens.push({ kind: "plain", text: source.slice(cursor, index) });
    }

    const [text, comment, tag, quoted, close] = match;
    if (comment) tokens.push({ kind: "comment", text });
    else if (tag) tokens.push({ kind: "tag", text });
    else if (quoted) tokens.push({ kind: "string", text });
    else if (close) tokens.push({ kind: "punctuation", text });

    cursor = index + text.length;
  }

  if (cursor < source.length) {
    tokens.push({ kind: "plain", text: source.slice(cursor) });
  }

  return tokens;
}

export const TOKEN_CLASS: Record<TokenKind, string> = {
  plain: "text-ink",
  comment: "text-ink-faint italic",
  string: "text-[#8ddb8d]",
  number: "text-[#e0a86a]",
  keyword: "text-[#c792ea]",
  literal: "text-[#7fb3ff]",
  function: "text-[#79c0ff]",
  punctuation: "text-ink-muted",
  tag: "text-[#f0868a]",
  property: "text-[#7fb3ff]",
};
