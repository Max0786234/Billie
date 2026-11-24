import { useCallback, useRef, useState } from "react";
import { api } from "../api/client";
import type {
  Analysis,
  DoneEvent,
  EmergencyAlert,
  Enrichment,
  Mode,
  RetrievalInfo,
  Source,
  Stage,
  StoredMessage,
} from "../api/types";

export interface UIMessage {
  key: string;
  messageId?: number;
  feedback?: number | null;
  role: "user" | "assistant";
  content: string;
  mode: Mode;
  imageIds?: string[];
  // assistant-only
  sources?: Source[];
  stage?: Stage | null;
  streaming?: boolean;
  error?: string;
  done?: Partial<DoneEvent>;
  analysis?: Analysis;
  retrieval?: RetrievalInfo;
  enrichment?: Enrichment;
  imageDescription?: string | null;
  alert?: EmergencyAlert | null;
}

let counter = 0;
const nextKey = () => `m${Date.now()}-${counter++}`;

export function fromStored(messages: StoredMessage[]): UIMessage[] {
  return messages.map((m) => ({
    key: `s${m.id}`,
    messageId: m.id,
    feedback: m.feedback,
    role: m.role,
    content: m.content,
    mode: m.mode,
    imageIds: m.meta?.image_ids,
    sources: m.sources,
    done: m.role === "assistant" ? m.meta : undefined,
    analysis: m.meta?.analysis,
    imageDescription: m.meta?.image_description ?? null,
    enrichment: m.meta?.enrichment,
    alert: m.meta?.alert ?? null,
  }));
}

export function useChat(onConversationChange: (id: string) => void) {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const patch = useCallback((key: string, update: Partial<UIMessage> | ((m: UIMessage) => Partial<UIMessage>)) => {
    setMessages((prev) =>
      prev.map((m) => (m.key === key ? { ...m, ...(typeof update === "function" ? update(m) : update) } : m)),
    );
  }, []);

  const send = useCallback(
    async (question: string, mode: Mode, imageIds: string[] = [], regenerate = false) => {
      if (busy || !question.trim()) return;
      const assistantKey = nextKey();
      setMessages((prev) => [
        // Regenerate / edit replaces the last question-answer pair instead of appending.
        ...(regenerate ? prev.slice(0, -2) : prev),
        { key: nextKey(), role: "user", content: question, mode, imageIds },
        { key: assistantKey, role: "assistant", content: "", mode, streaming: true, stage: "retrieving", sources: [] },
      ]);
      setBusy(true);
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const stream = api.chat(
          { question, mode, conversation_id: conversationId, image_ids: imageIds, regenerate },
          controller.signal,
        );
        for await (const event of stream) {
          switch (event.event) {
            case "status":
              patch(assistantKey, { stage: event.data.stage });
              break;
            case "meta":
              if (event.data.conversation_id !== conversationId) {
                setConversationId(event.data.conversation_id);
              }
              patch(assistantKey, {
                sources: event.data.sources,
                analysis: event.data.analysis,
                retrieval: event.data.retrieval,
                enrichment: event.data.enrichment,
                imageDescription: event.data.image_description,
                alert: event.data.alert,
              });
              break;
            case "token":
              patch(assistantKey, (m) => ({ content: m.content + event.data.text }));
              break;
            case "done":
              patch(assistantKey, {
                content: event.data.answer,
                done: event.data,
                messageId: event.data.message_id,
                streaming: false,
                stage: null,
              });
              onConversationChange(event.data.conversation_id);
              break;
            case "error":
              patch(assistantKey, { error: event.data.message, streaming: false, stage: null });
              break;
          }
        }
      } catch (err) {
        const aborted = controller.signal.aborted;
        patch(assistantKey, (m) => ({
          streaming: false,
          stage: null,
          error: aborted ? undefined : (err as Error).message,
          content: aborted && m.content ? `${m.content}\n\n_(stopped)_` : m.content || (aborted ? "_(stopped)_" : ""),
        }));
      } finally {
        abortRef.current = null;
        setBusy(false);
      }
    },
    [busy, conversationId, onConversationChange, patch],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  /** Ask the last question again (or an edited version of it), replacing the last answer. */
  const regenerate = useCallback(
    (newQuestion?: string) => {
      const lastUser = [...messages].reverse().find((m) => m.role === "user");
      if (!lastUser || messages[messages.length - 1]?.role !== "assistant") return;
      void send(newQuestion ?? lastUser.content, lastUser.mode, lastUser.imageIds ?? [], true);
    },
    [messages, send],
  );

  const rate = useCallback(
    async (key: string, messageId: number, value: 1 | -1) => {
      const current = messages.find((m) => m.key === key)?.feedback ?? null;
      const next = current === value ? 0 : value; // clicking again clears
      patch(key, { feedback: next || null });
      try {
        await api.feedback(messageId, next as 1 | -1 | 0);
      } catch {
        patch(key, { feedback: current });
      }
    },
    [messages, patch],
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setConversationId(null);
  }, []);

  const load = useCallback(async (id: string) => {
    abortRef.current?.abort();
    const conversation = await api.conversation(id);
    setConversationId(id);
    setMessages(fromStored(conversation.messages));
    return conversation;
  }, []);

  return { messages, conversationId, busy, send, stop, reset, load, regenerate, rate };
}
