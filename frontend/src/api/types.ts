// Mirrors the payloads produced by backend/src/billie/chat/pipeline.py and api/app.py.

export type Mode = "general" | "medical" | "research";

export interface Source {
  n: number;
  id: string;
  title: string;
  source: string;
  url: string;
  text: string;
  score: number;
  components: Record<string, number>;
  metadata: Record<string, unknown>;
}

export interface Analysis {
  language: string;
  language_name: string;
  language_confidence: number;
  sentiment: "positive" | "neutral" | "negative";
  sentiment_score: number;
  method: Record<string, string>;
}

export interface RetrievalInfo {
  query: string;
  translated_query: string | null;
  mode: string;
  degraded: boolean;
  note: string;
  timings_ms: Record<string, number>;
}

export interface Enrichment {
  entities?: { text: string; category: string }[];
  disclaimer?: string;
  concepts?: { text: string; score: number }[];
}

export interface EmergencyAlert {
  categories: string[];
  message: string;
}

export interface MetaEvent {
  conversation_id: string;
  mode: Mode;
  question: string;
  analysis: Analysis;
  sources: Source[];
  retrieval: RetrievalInfo;
  image_description: string | null;
  enrichment: Enrichment;
  alert: EmergencyAlert | null;
}

export interface DoneEvent {
  conversation_id: string;
  message_id: number;
  answer: string;
  cited_sources: Source[];
  analysis: Analysis;
  grounded: boolean | null;
  abstained: boolean;
  cached?: boolean;
  not_found?: boolean;
  auto_citations?: { sentence: string; n: number; score: number }[];
  unverified_sentences?: number;
  /** Dead-end guidance: closest topics in this mode, and a better-suited mode if one clearly fits. */
  related?: Suggestion[];
  suggested_mode?: { mode: Mode; score: number } | null;
  citations: number[];
  invalid_citations: number[];
  used_llm: boolean;
  provider: string | null;
  model: string | null;
  retrieval_mode: string;
  translated_query?: string | null;
  timings_ms: Record<string, number>;
  image_ids: string[];
  image_description: string | null;
  enrichment?: Enrichment;
  alert?: EmergencyAlert | null;
  notes: string[];
}

export type Stage = "reading_image" | "translating" | "retrieving" | "generating";

export type ChatEvent =
  | { event: "status"; data: { stage: Stage } }
  | { event: "meta"; data: MetaEvent }
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: DoneEvent }
  | { event: "error"; data: { message: string } };

export interface ConversationSummary {
  id: string;
  title: string;
  mode: Mode;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface StoredMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  mode: Mode;
  sources: Source[];
  meta: Partial<DoneEvent> & { image_ids?: string[] };
  created_at: number;
  feedback: number | null;
}

export interface Conversation extends ConversationSummary {
  messages: StoredMessage[];
}

export interface LLMSettings {
  provider: string;
  model: string;
  base_url: string;
  vision_model: string;
  temperature: number;
  max_tokens: number;
  keys: Record<string, string | null>;
}

export interface ProviderPreset {
  id: string;
  label: string;
  base_url: string;
  default_model: string;
  suggested_models: string[];
  needs_key: boolean;
  key_url: string | null;
}

export interface CorpusStatus {
  name: "knowledge" | "medquad" | "arxiv";
  ready: boolean;
  variant: string;
  passages: number;
  vectors: number;
  embedding_model: string | null;
  built_at: string | null;
  search_mode: string;
}

export interface SystemStatus {
  indexing: boolean;
  llm: { ok: boolean; provider: string; model: string; detail: string; models?: string[] };
  embedding: { model: string | null; available: boolean };
  corpora: CorpusStatus[];
  cache: { entries: number; hits: number } | null;
  feedback: Record<string, { up: number; down: number }>;
  environment: string;
  auth_required: boolean;
  /** Public demo quotas; null when running locally. */
  demo: { daily_limit: number; used_today: number; per_visitor_limit: number; remaining_for_you: number } | null;
}

export interface DocumentInfo {
  name: string;
  size: number;
  modified: number;
}

/** A question the user can click: a real title from the index and the question to send for it. */
export interface Suggestion {
  title: string;
  question: string;
}

export interface NamedCount {
  name: string;
  count: number;
}

/** What one mode can answer, computed from its index (GET /api/overview). */
export interface ModeOverview {
  corpus: string;
  passages: number;
  vectors: number;
  questions?: number;
  topics?: number;
  sources?: NamedCount[];
  question_types?: NamedCount[];
  years?: [number, number] | null;
  categories?: NamedCount[];
  documents?: NamedCount[];
  good_for: string[];
  not_covered: string[];
  examples: string[];
}

export interface Overview {
  modes: Record<Mode, ModeOverview>;
}
