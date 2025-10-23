import { Buffer } from "buffer";

const STORAGE_KEY = "lexiai.session";

export type Session = {
  tenantId: string;
  userId: string;
  roles: string[];
  token: string;
  expiresAt?: string | null;
};

let cachedSession: Session | null = null;

function decodeBase64Url(value: string): string {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const padding = normalized.length % 4 === 0 ? "" : "=".repeat(4 - (normalized.length % 4));
  if (typeof window === "undefined") {
    return Buffer.from(`${normalized}${padding}`, "base64").toString("utf-8");
  }
  const binary = window.atob(`${normalized}${padding}`);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new TextDecoder().decode(bytes);
}

function parseToken(token: string): Session | null {
  const parts = token.split(".");
  if (parts.length < 2) {
    return null;
  }
  try {
    const payloadJson = decodeBase64Url(parts[1]);
    const payload = JSON.parse(payloadJson) as {
      tenant?: string;
      sub?: string;
      roles?: string[];
      exp?: number;
    };
    if (!payload.tenant || !payload.sub) {
      return null;
    }
    return {
      tenantId: payload.tenant,
      userId: payload.sub,
      roles: Array.isArray(payload.roles) ? payload.roles : [],
      token,
      expiresAt: payload.exp ? new Date(payload.exp * 1000).toISOString() : null,
    };
  } catch (error) {
    console.error("Failed to parse token", error);
    return null;
  }
}

function readStorage(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch (error) {
    console.warn("Unable to access localStorage", error);
    return null;
  }
}

export function loadSession(): Session | null {
  if (cachedSession) {
    return cachedSession;
  }
  const stored = readStorage();
  if (!stored) {
    return null;
  }
  try {
    const parsed = JSON.parse(stored) as { token?: string };
    if (!parsed.token) {
      return null;
    }
    const session = parseToken(parsed.token);
    cachedSession = session;
    return session;
  } catch (error) {
    console.error("Invalid session payload", error);
    return null;
  }
}

export function storeSession(token: string): Session {
  const session = parseToken(token);
  if (!session) {
    throw new Error("Token payload is invalid or missing tenant information");
  }
  cachedSession = session;
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ token }));
    } catch (error) {
      console.error("Failed to persist session", error);
    }
  }
  return session;
}

export function clearSession(): void {
  cachedSession = null;
  if (typeof window !== "undefined") {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error("Failed to clear session", error);
    }
  }
}

export function getSession(): Session | null {
  if (typeof window === "undefined") {
    return cachedSession;
  }
  return loadSession();
}

export function getBearerToken(): string | null {
  const session = getSession();
  return session?.token ?? null;
}

export { parseToken };
