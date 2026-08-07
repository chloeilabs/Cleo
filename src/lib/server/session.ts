import "server-only";

import {
  createCipheriv,
  createDecipheriv,
  createHash,
  randomBytes,
} from "node:crypto";

import { cookies } from "next/headers";

import { Cursor } from "@cursor/sdk";

import type { SessionState, SessionUser } from "@/lib/types";

export const SESSION_COOKIE = "cleo_session";

const ALGORITHM = "aes-256-gcm";
const IV_BYTES = 12;

/**
 * Cookies are encrypted with `CLEO_SESSION_SECRET` when it is configured, which
 * lets sessions survive restarts and span multiple server instances. Without
 * it we fall back to a per-process key: still safe, but everyone has to sign in
 * again after a restart.
 */
const secretMaterial =
  process.env.CLEO_SESSION_SECRET ?? randomBytes(32).toString("hex");
const secretKey = createHash("sha256").update(secretMaterial).digest();

export class MissingApiKeyError extends Error {
  readonly code = "missing_api_key";

  constructor(message = "Sign in with a Cursor API key to continue.") {
    super(message);
    this.name = "MissingApiKeyError";
  }
}

export class InvalidApiKeyError extends Error {
  readonly code = "invalid_api_key";

  constructor(message = "That Cursor API key could not be verified.") {
    super(message);
    this.name = "InvalidApiKeyError";
  }
}

function seal(value: string): string {
  const iv = randomBytes(IV_BYTES);
  const cipher = createCipheriv(ALGORITHM, secretKey, iv);
  const encrypted = Buffer.concat([
    cipher.update(value, "utf8"),
    cipher.final(),
  ]);
  return [
    iv.toString("base64url"),
    cipher.getAuthTag().toString("base64url"),
    encrypted.toString("base64url"),
  ].join(".");
}

function open(sealed: string): string | undefined {
  const [ivPart, tagPart, dataPart] = sealed.split(".");
  if (!ivPart || !tagPart || !dataPart) return undefined;

  try {
    const decipher = createDecipheriv(
      ALGORITHM,
      secretKey,
      Buffer.from(ivPart, "base64url"),
    );
    decipher.setAuthTag(Buffer.from(tagPart, "base64url"));
    return Buffer.concat([
      decipher.update(Buffer.from(dataPart, "base64url")),
      decipher.final(),
    ]).toString("utf8");
  } catch {
    return undefined;
  }
}

export function environmentApiKey(): string | undefined {
  const key = process.env.CURSOR_API_KEY?.trim();
  return key ? key : undefined;
}

/** Resolve the caller's API key, or `undefined` when not signed in. */
export async function readApiKey(): Promise<string | undefined> {
  const sealed = (await cookies()).get(SESSION_COOKIE)?.value;
  if (sealed) {
    const opened = open(sealed)?.trim();
    if (opened) return opened;
  }
  return environmentApiKey();
}

/** Resolve the caller's API key or throw, for routes that require auth. */
export async function requireApiKey(): Promise<string> {
  const apiKey = await readApiKey();
  if (!apiKey) throw new MissingApiKeyError();
  return apiKey;
}

export function toSessionUser(user: {
  apiKeyName: string;
  userId?: number;
  userEmail?: string;
  userFirstName?: string;
  userLastName?: string;
  createdAt: string;
}): SessionUser {
  const fullName = [user.userFirstName, user.userLastName]
    .filter(Boolean)
    .join(" ")
    .trim();

  return {
    apiKeyName: user.apiKeyName,
    userId: user.userId,
    email: user.userEmail,
    firstName: user.userFirstName,
    lastName: user.userLastName,
    displayName: fullName || user.userEmail || user.apiKeyName,
    createdAt: user.createdAt,
  };
}

export async function verifyApiKey(apiKey: string): Promise<SessionUser> {
  const trimmed = apiKey.trim();
  if (!trimmed) {
    throw new InvalidApiKeyError("Enter a Cursor API key.");
  }

  try {
    return toSessionUser(await Cursor.me({ apiKey: trimmed }));
  } catch {
    throw new InvalidApiKeyError(
      "That key was rejected by Cursor. Create one at cursor.com/dashboard/api and try again.",
    );
  }
}

export async function startSession(apiKey: string): Promise<SessionUser> {
  const user = await verifyApiKey(apiKey);

  (await cookies()).set(SESSION_COOKIE, seal(apiKey.trim()), {
    httpOnly: true,
    sameSite: "strict",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });

  return user;
}

export async function endSession(): Promise<void> {
  (await cookies()).delete(SESSION_COOKIE);
}

export async function currentSession(): Promise<SessionState> {
  const apiKey = await readApiKey();
  if (!apiKey) {
    return { authenticated: false, fromEnvironment: false };
  }

  const fromEnvironment =
    !(await cookies()).get(SESSION_COOKIE)?.value && Boolean(environmentApiKey());

  try {
    return {
      authenticated: true,
      fromEnvironment,
      user: toSessionUser(await Cursor.me({ apiKey })),
    };
  } catch {
    return { authenticated: false, fromEnvironment };
  }
}
