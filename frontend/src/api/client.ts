import { readEventStream } from "./sse";
import type {
  ChatEvent,
  Conversation,
  ConversationSummary,
  DocumentInfo,
  LLMSettings,
  Mode,
  Overview,
  ProviderPreset,
  Suggestion,
  SystemStatus,
} from "./types";

const KEY_STORAGE = "billie.keys";

export interface StoredKeys {
  apiKey?: string;
  adminKey?: string;
}

export function loadKeys(): StoredKeys {
  try {
    return JSON.parse(localStorage.getItem(KEY_STORAGE) ?? "{}") as StoredKeys;
  } catch {
    return {};
  }
}

export function saveKeys(keys: StoredKeys) {
  localStorage.setItem(KEY_STORAGE, JSON.stringify(keys));
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

const VISITOR_STORAGE = "billie.visitor";

/** Random per-browser id. On the public demo the server keeps each visitor's chats separate by it. */
export function visitorId(): string {
  try {
    let id = localStorage.getItem(VISITOR_STORAGE);
    if (!id || !/^[A-Za-z0-9-]{16,64}$/.test(id)) {
      id = crypto.randomUUID();
      localStorage.setItem(VISITOR_STORAGE, id);
    }
    return id;
  } catch {
    return ""; // storage blocked: the server falls back to the client IP
  }
}

function headers(extra?: HeadersInit): Headers {
  const h = new Headers(extra);
  const visitor = visitorId();
  if (visitor) h.set("X-Visitor-Id", visitor);
  const keys = loadKeys();
  if (keys.apiKey) h.set("X-API-Key", keys.apiKey);
  if (keys.adminKey) h.set("X-Admin-Key", keys.adminKey);
  return h;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { ...init, headers: headers(init.headers) });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  body: JSON.stringify(body),
  headers: { "Content-Type": "application/json" },
});

export const api = {
  status: () => request<SystemStatus>("/api/status"),
  overview: () => request<Overview>("/api/overview"),
  suggest: (mode: Mode, q: string, signal?: AbortSignal) =>
    request<Suggestion[]>(`/api/suggest?mode=${mode}&q=${encodeURIComponent(q)}`, { signal }),

  conversations: () => request<ConversationSummary[]>("/api/conversations"),
  conversation: (id: string) => request<Conversation>(`/api/conversations/${id}`),
  renameConversation: (id: string, title: string) =>
    request(`/api/conversations/${id}`, { method: "PATCH", ...json({ title }) }),
  deleteConversation: (id: string) => request(`/api/conversations/${id}`, { method: "DELETE" }),

  feedback: (messageId: number, value: 1 | -1 | 0) =>
    request(`/api/messages/${messageId}/feedback`, { method: "PUT", ...json({ value }) }),

  uploadImage: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ id: string; media_type: string; size: number }>("/api/uploads/images", {
      method: "POST",
      body: form,
    });
  },
  imageUrl: (id: string) => `/api/uploads/images/${id}`,

  documents: () => request<DocumentInfo[]>("/api/documents"),
  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentInfo>("/api/documents", { method: "POST", body: form });
  },
  deleteDocument: (name: string) => request(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" }),

  llmSettings: () => request<LLMSettings>("/api/settings/llm"),
  updateLlmSettings: (patch: Partial<Omit<LLMSettings, "keys">> & { api_key?: string }) =>
    request<LLMSettings>("/api/settings/llm", { method: "PUT", ...json(patch) }),
  presets: () => request<ProviderPreset[]>("/api/llm/presets"),
  models: () => request<{ models: string[]; error?: string }>("/api/llm/models"),
  testLlm: (body: { provider?: string; model?: string; base_url?: string; api_key?: string }) =>
    request<{ ok: boolean; detail: string; models?: string[] }>("/api/llm/test", { method: "POST", ...json(body) }),

  async *chat(
    body: { question: string; mode: Mode; conversation_id?: string | null; image_ids?: string[]; regenerate?: boolean },
    signal?: AbortSignal,
  ): AsyncGenerator<ChatEvent> {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify(body),
      signal,
    });
    if (!response.ok || !response.body) {
      let detail = response.statusText;
      try {
        const err = await response.json();
        detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
      } catch {
        /* ignore */
      }
      throw new ApiError(detail, response.status);
    }
    yield* readEventStream(response.body);
  },
};
