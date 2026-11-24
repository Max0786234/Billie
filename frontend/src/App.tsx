import { AlertTriangle, PanelLeftOpen } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api/client";
import type { ConversationSummary, LLMSettings, Mode, Overview, SystemStatus } from "./api/types";
import { Composer } from "./components/Composer";
import { DocumentsDialog, PrivateDocumentsDialog } from "./components/DocumentsDialog";
import { ModeGuide } from "./components/ModeGuide";
import { AssistantMessage, UserMessage } from "./components/MessageView";
import { SettingsDialog } from "./components/SettingsDialog";
import { Sidebar } from "./components/Sidebar";
import { StatusDialog } from "./components/StatusDialog";
import { useChat } from "./hooks/useChat";
import { cx } from "./lib/format";
import { MODES } from "./lib/modes";

type DialogName = "settings" | "documents" | "shared-documents" | "status" | null;

function initialDark(): boolean {
  const stored = localStorage.getItem("billie.theme");
  return stored ? stored === "dark" : window.matchMedia?.("(prefers-color-scheme: dark)").matches;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("general");
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [dialog, setDialog] = useState<DialogName>(null);
  const [dark, setDark] = useState(initialDark);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [llm, setLlm] = useState<LLMSettings | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  const refreshConversations = useCallback(() => {
    api.conversations().then(setConversations).catch((e: Error) => setLoadError(e.message));
  }, []);

  const isDemo = useRef(false);
  isDemo.current = Boolean(status?.demo);
  const afterTurn = useCallback(() => {
    refreshConversations();
    if (isDemo.current) api.status().then(setStatus).catch(() => undefined); // update "questions left"
  }, [refreshConversations]);

  const chat = useChat(afterTurn);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("billie.theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    refreshConversations();
    api.llmSettings().then(setLlm).catch(() => undefined);
    api.status().then(setStatus).catch((e: Error) => setLoadError(e.message));
    api.overview().then(setOverview).catch(() => undefined); // the guide falls back to built-in text
    // Deep link: ?c=<conversation id> reopens a saved conversation.
    const linked = new URLSearchParams(window.location.search).get("c");
    if (linked && /^[0-9a-f]{32}$/.test(linked)) {
      chat.load(linked).then((c) => setMode(c.mode)).catch(() => window.history.replaceState(null, "", "/"));
    }
  }, [refreshConversations]);

  // Keep the URL in sync with the open conversation so it can be bookmarked or reloaded.
  useEffect(() => {
    const target = chat.conversationId ? `/?c=${chat.conversationId}` : "/";
    if (window.location.pathname + window.location.search !== target) window.history.replaceState(null, "", target);
  }, [chat.conversationId]);

  // Follow the streaming answer unless the user scrolled up to read.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [chat.messages]);

  const startNew = (nextMode: Mode = mode) => {
    chat.reset();
    setMode(nextMode);
    setSidebarOpen(false);
  };

  const openConversation = async (id: string) => {
    const conversation = await chat.load(id);
    setMode(conversation.mode);
    setSidebarOpen(false);
  };

  /** Follow a suggestion from an answer: same mode, or start over in a better-suited mode. */
  const ask = (question: string | undefined, target: Mode | undefined, previous: string | undefined) => {
    const text = question ?? previous;
    if (!text) return;
    stickToBottom.current = true;
    if (target && target !== mode) {
      chat.reset();
      setMode(target);
      void chat.send(text, target);
    } else {
      void chat.send(text, mode);
    }
  };

  const info = MODES[mode];
  const llmDown = status && !status.llm.ok;

  return (
    <div className="flex h-full">
      <div
        className={cx(
          "fixed inset-y-0 left-0 z-40 transition-transform lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <Sidebar
          mode={mode}
          onModeChange={(m) => (chat.messages.length ? startNew(m) : setMode(m))}
          conversations={conversations}
          activeId={chat.conversationId}
          onNew={() => startNew()}
          onOpen={(id) => void openConversation(id)}
          onRename={(id, title) => void api.renameConversation(id, title).then(refreshConversations)}
          onDelete={(id) => {
            void api.deleteConversation(id).then(() => {
              if (id === chat.conversationId) chat.reset();
              refreshConversations();
            });
          }}
          onOpenSettings={() => setDialog("settings")}
          onOpenDocuments={() => setDialog("documents")}
          onOpenStatus={() => setDialog("status")}
          dark={dark}
          onToggleDark={() => setDark((d) => !d)}
          onClose={() => setSidebarOpen(false)}
        />
      </div>
      {sidebarOpen && <div className="fixed inset-0 z-30 bg-black/30 lg:hidden" onClick={() => setSidebarOpen(false)} />}

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-800">
          <button
            type="button"
            className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-100 lg:hidden dark:hover:bg-zinc-800"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open sidebar"
          >
            <PanelLeftOpen className="size-5" />
          </button>
          <info.icon className={cx("size-5", info.accent.text)} />
          <h1 className="font-medium">{info.label}</h1>
          <span className="hidden truncate text-sm text-zinc-500 sm:inline">{info.tagline}</span>
          <button
            type="button"
            onClick={() => setDialog("settings")}
            className="ml-auto flex items-center gap-1.5 rounded-full border border-zinc-200 px-2.5 py-1 text-xs text-zinc-600 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
            title="Model settings"
          >
            <span className={cx("size-1.5 rounded-full", llmDown ? "bg-red-500" : "bg-emerald-500")} />
            {llm ? `${llm.provider} · ${llm.model}` : "model"}
          </button>
        </header>

        {(llmDown || loadError) && (
          <div className="flex items-center gap-2 border-b border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300">
            <AlertTriangle className="size-4 shrink-0" />
            {loadError
              ? `Cannot reach the Billie API (${loadError}). Is the backend running on port 8000?`
              : `The language model is unavailable: ${status?.llm.detail}. Answers will show retrieved passages only.`}
          </div>
        )}

        {status?.demo && (
          <div className="border-b border-sky-200 bg-sky-50 px-4 py-1.5 text-center text-xs text-sky-800 dark:border-sky-900 dark:bg-sky-950/40 dark:text-sky-300">
            Public demo · {status.demo.remaining_for_you} of {status.demo.per_visitor_limit} questions left for you today
            · chats are private to this browser ·{" "}
            <a className="underline" href="https://github.com/Vishaljhaaa/Billie" target="_blank" rel="noreferrer">
              run it locally
            </a>{" "}
            for unlimited use
          </div>
        )}

        <div
          ref={scrollRef}
          className="min-h-0 flex-1 overflow-y-auto"
          onScroll={(e) => {
            const el = e.currentTarget;
            stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}
        >
          <div className="mx-auto max-w-3xl px-4 py-6">
            {chat.messages.length === 0 ? (
              <ModeGuide
                mode={mode}
                overview={overview?.modes[mode]}
                onPick={(q) => void chat.send(q, mode)}
                onOpenDocuments={() => setDialog("documents")}
                privateUploads={Boolean(status?.demo)}
              />
            ) : (
              <div className="space-y-7">
                {chat.messages.map((m, i) => {
                  const isLast = i === chat.messages.length - 1;
                  const isLastQuestion = i === chat.messages.length - 2 && m.role === "user";
                  return m.role === "user" ? (
                    <UserMessage key={m.key} message={m} onEdit={isLastQuestion && !chat.busy ? (t) => chat.regenerate(t) : undefined} />
                  ) : (
                    <AssistantMessage
                      key={m.key}
                      message={m}
                      onAsk={isLast && !chat.busy ? (q, target) => ask(q, target, chat.messages[i - 1]?.content) : undefined}
                      onRegenerate={isLast && !chat.busy ? () => chat.regenerate() : undefined}
                      onRate={m.messageId !== undefined ? (v) => void chat.rate(m.key, m.messageId!, v) : undefined}
                    />
                  );
                })}
              </div>
            )}
          </div>
        </div>

        <div className="mx-auto w-full max-w-3xl px-4 pb-4">
          <Composer
            mode={mode}
            busy={chat.busy}
            onSend={(text, images) => {
              stickToBottom.current = true;
              void chat.send(text, mode, images);
            }}
            onStop={chat.stop}
          />
          <p className="mt-2 text-center text-[11px] text-zinc-400">
            Billie answers from {info.corpus} and cites its sources. It can still make mistakes - check the citations.
          </p>
        </div>
      </main>

      {dialog === "settings" && <SettingsDialog onClose={() => setDialog(null)} onSaved={(s) => {
        setLlm(s);
        api.status().then(setStatus).catch(() => undefined);
      }} />}
      {dialog === "documents" &&
        (status?.demo ? (
          <PrivateDocumentsDialog onClose={() => setDialog(null)} onAdmin={() => setDialog("shared-documents")} />
        ) : (
          <DocumentsDialog onClose={() => setDialog(null)} />
        ))}
      {dialog === "shared-documents" && <DocumentsDialog onClose={() => setDialog(null)} />}
      {dialog === "status" && <StatusDialog onClose={() => setDialog(null)} />}
    </div>
  );
}
