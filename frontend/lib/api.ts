import { API_BASE_URL } from "./config";
import { getBearerToken } from "./session";

type RequestOptions = RequestInit & { skipJsonParsing?: boolean };

async function handleResponse<T>(response: Response, skipJsonParsing?: boolean): Promise<T> {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }
  if (skipJsonParsing) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

function ensureHeaders(
  headers: RequestInit["headers"],
  body: BodyInit | null | undefined,
  defaults: HeadersInit,
): HeadersInit | undefined {
  const isFormData = typeof FormData !== "undefined" && body instanceof FormData;
  const merged = new Headers(defaults);

  if (headers instanceof Headers) {
    headers.forEach((value, key) => merged.set(key, value));
  } else if (Array.isArray(headers)) {
    headers.forEach(([key, value]) => merged.set(key, value));
  } else if (headers) {
    Object.entries(headers).forEach(([key, value]) => {
      if (typeof value !== "undefined") {
        merged.set(key, String(value));
      }
    });
  }

  if (isFormData) {
    merged.delete("Content-Type");
    return merged;
  }

  if (!merged.has("Content-Type")) {
    merged.set("Content-Type", "application/json");
  }

  return merged;
}

export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { skipJsonParsing, headers, body, ...rest } = options;
  const token = getBearerToken();
  const authHeader: HeadersInit = token ? { Authorization: `Bearer ${token}` } : {};
  const finalHeaders = ensureHeaders(headers, body, authHeader);
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...rest,
    body,
    headers: finalHeaders,
  });
  return handleResponse<T>(response, skipJsonParsing);
}
