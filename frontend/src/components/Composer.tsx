import { ArrowUp, ImagePlus, Loader2, Search, Square, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Mode, Suggestion } from "../api/types";
import { cx } from "../lib/format";
import { MODES } from "../lib/modes";

interface Attachment {
  id: string;
  preview: string;
}

interface Props {
  mode: Mode;
  busy: boolean;
  onSend: (text: string, imageIds: string[]) => void;
  onStop: () => void;
}

const MAX_IMAGES = 4;

export function Composer({ mode, busy, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [active, setActive] = useState(-1);
  const [dismissed, setDismissed] = useState<string | null>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const el = textRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);

  useEffect(() => textRef.current?.focus(), [mode]);

  // As-you-type suggestions: real titles from this mode's index, so each one is answerable.
  useEffect(() => {
    const query = text.trim();
    if (busy || query.length < 2 || text.length > 120 || text.includes("\n") || dismissed === text) {
      setSuggestions([]);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      api
        .suggest(mode, query, controller.signal)
        .then((found) => {
          setSuggestions(found);
          setActive(-1);
        })
        .catch(() => undefined); // aborted or offline: no suggestions, typing still works
    }, 180);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [text, mode, busy, dismissed]);

  const addFiles = async (files: Iterable<File>) => {
    const images = [...files].filter((f) => f.type.startsWith("image/"));
    if (!images.length) return;
    setError(null);
    setUploading(true);
    try {
      for (const file of images.slice(0, MAX_IMAGES - attachments.length)) {
        const stored = await api.uploadImage(file);
        setAttachments((prev) => [...prev, { id: stored.id, preview: URL.createObjectURL(file) }]);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const send = (question: string) => {
    if (busy || uploading || !question.trim()) return;
    onSend(question.trim(), attachments.map((a) => a.id));
    attachments.forEach((a) => URL.revokeObjectURL(a.preview));
    setText("");
    setAttachments([]);
    setSuggestions([]);
  };
  const submit = () => send(text);

  const accent = MODES[mode].accent;

  return (
    <div
      className={cx(
        "relative rounded-2xl border bg-white shadow-sm transition dark:bg-zinc-900",
        dragging ? "border-zinc-400 ring-4 ring-zinc-200 dark:ring-zinc-800" : "border-zinc-200 dark:border-zinc-800",
      )}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        void addFiles(e.dataTransfer.files);
      }}
    >
      {suggestions.length > 0 && (
        <div
          role="listbox"
          aria-label="Suggested questions"
          className="absolute right-0 bottom-full left-0 z-20 mb-2 overflow-hidden rounded-xl border border-zinc-200 bg-white text-left shadow-lg dark:border-zinc-700 dark:bg-zinc-900"
        >
          <div className="flex items-center gap-1.5 px-3 pt-2 pb-1 text-[11px] font-medium tracking-wide text-zinc-500 uppercase">
            <Search className="size-3" /> In the {MODES[mode].corpus} index
          </div>
          {suggestions.map((s, i) => (
            <button
              key={s.title}
              type="button"
              role="option"
              aria-selected={i === active}
              onMouseDown={(e) => e.preventDefault()} // keep focus in the text box
              onClick={() => send(s.question)}
              className={cx(
                "block w-full truncate px-3 py-1.5 text-left text-sm",
                i === active ? "bg-zinc-100 dark:bg-zinc-800" : "hover:bg-zinc-50 dark:hover:bg-zinc-800/60",
              )}
              title={s.question}
            >
              {s.title}
            </button>
          ))}
        </div>
      )}
      {(attachments.length > 0 || uploading) && (
        <div className="flex flex-wrap gap-2 px-3 pt-3">
          {attachments.map((a) => (
            <div key={a.id} className="group relative">
              <img src={a.preview} alt="Attachment" className="size-16 rounded-lg object-cover" />
              <button
                type="button"
                aria-label="Remove image"
                onClick={() => setAttachments((prev) => prev.filter((p) => p.id !== a.id))}
                className="absolute -top-1.5 -right-1.5 rounded-full bg-zinc-900 p-0.5 text-white opacity-0 transition group-hover:opacity-100"
              >
                <X className="size-3" />
              </button>
            </div>
          ))}
          {uploading && (
            <div className="flex size-16 items-center justify-center rounded-lg bg-zinc-100 dark:bg-zinc-800">
              <Loader2 className="size-4 animate-spin text-zinc-500" />
            </div>
          )}
        </div>
      )}
      <textarea
        ref={textRef}
        rows={1}
        value={text}
        placeholder={`Ask in ${MODES[mode].label} mode…`}
        aria-label="Message"
        onChange={(e) => setText(e.target.value)}
        onPaste={(e) => {
          const files = [...e.clipboardData.files];
          if (files.length) {
            e.preventDefault();
            void addFiles(files);
          }
        }}
        onKeyDown={(e) => {
          if (suggestions.length) {
            if (e.key === "ArrowDown" || e.key === "ArrowUp") {
              e.preventDefault();
              const last = suggestions.length - 1;
              // -1 = nothing selected (Enter sends what was typed).
              setActive((a) => (e.key === "ArrowDown" ? (a >= last ? -1 : a + 1) : a <= -1 ? last : a - 1));
              return;
            }
            if (e.key === "Escape") {
              setDismissed(text);
              return;
            }
            if (e.key === "Enter" && active >= 0 && active < suggestions.length) {
              e.preventDefault();
              send(suggestions[active]?.question ?? text);
              return;
            }
          }
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
        className="block max-h-56 w-full resize-none bg-transparent px-4 pt-3.5 pb-2 text-[15px] outline-none placeholder:text-zinc-400"
      />
      <div className="flex items-center justify-between px-2.5 pb-2.5">
        <div className="flex items-center gap-1">
          <input
            ref={fileRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files) void addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={attachments.length >= MAX_IMAGES}
            className="rounded-lg p-2 text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-800 disabled:opacity-40 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
            title="Attach an image (or paste / drop one)"
          >
            <ImagePlus className="size-5" />
          </button>
          {error && <span className="text-xs text-red-600">{error}</span>}
        </div>
        {busy ? (
          <button
            type="button"
            onClick={onStop}
            className="flex size-9 items-center justify-center rounded-full bg-zinc-900 text-white transition hover:bg-zinc-700 dark:bg-zinc-100 dark:text-zinc-900"
            title="Stop"
          >
            <Square className="size-3.5 fill-current" />
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!text.trim() || uploading}
            className={cx(
              "flex size-9 items-center justify-center rounded-full text-white transition disabled:bg-zinc-300 dark:disabled:bg-zinc-700",
              accent.bg,
            )}
            title="Send (Enter)"
          >
            <ArrowUp className="size-5" />
          </button>
        )}
      </div>
    </div>
  );
}
