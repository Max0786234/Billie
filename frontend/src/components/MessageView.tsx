import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Copy,
  Info,
  Loader2,
  Pencil,
  Phone,
  RefreshCw,
  ShieldAlert,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Zap,
} from "lucide-react";
import { useRef, useState } from "react";
import { api } from "../api/client";
import type { Mode } from "../api/types";
import type { UIMessage } from "../hooks/useChat";
import { cx, formatMs } from "../lib/format";
import { MODES, stageLabel } from "../lib/modes";
import { Markdown } from "./Markdown";
import { SourceCard } from "./SourceCard";

interface UserProps {
  message: UIMessage;
  onEdit?: (text: string) => void;
}

export function UserMessage({ message, onEdit }: UserProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);
  if (editing && onEdit) {
    return (
      <form
        className="ml-auto flex max-w-[85%] flex-col gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (draft.trim()) onEdit(draft.trim());
          setEditing(false);
        }}
      >
        <textarea
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          rows={3}
          className="w-full resize-none rounded-2xl border border-zinc-300 bg-white px-4 py-2.5 outline-none dark:border-zinc-600 dark:bg-zinc-900"
        />
        <div className="flex justify-end gap-2 text-sm">
          <button type="button" onClick={() => setEditing(false)} className="rounded-lg px-3 py-1.5 hover:bg-zinc-100 dark:hover:bg-zinc-800">
            Cancel
          </button>
          <button type="submit" className="rounded-lg bg-zinc-900 px-3 py-1.5 text-white dark:bg-zinc-100 dark:text-zinc-900">
            Send
          </button>
        </div>
      </form>
    );
  }
  return (
    <div className="group flex justify-end">
      <div className="max-w-[85%] space-y-2">
        {message.imageIds?.length ? (
          <div className="flex flex-wrap justify-end gap-2">
            {message.imageIds.map((id) => (
              <img
                key={id}
                src={api.imageUrl(id)}
                alt="Attached"
                className="max-h-40 rounded-lg border border-zinc-200 object-cover dark:border-zinc-800"
              />
            ))}
          </div>
        ) : null}
        <div className="rounded-2xl rounded-br-md bg-zinc-100 px-4 py-2.5 whitespace-pre-wrap dark:bg-zinc-800">
          {message.content}
        </div>
        {onEdit && (
          <div className="flex justify-end opacity-0 transition group-hover:opacity-100">
            <button
              type="button"
              onClick={() => {
                setDraft(message.content);
                setEditing(true);
              }}
              className="inline-flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
            >
              <Pencil className="size-3" /> Edit
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

interface AssistantProps {
  message: UIMessage;
  onRegenerate?: () => void;
  onRate?: (value: 1 | -1) => void;
  /** Ask a follow-up: a given question, or (question undefined) the same question in another mode. */
  onAsk?: (question: string | undefined, mode: Mode | undefined) => void;
}

export function AssistantMessage({ message, onRegenerate, onRate, onAsk }: AssistantProps) {
  const sources = message.sources ?? [];
  const cited = new Set(message.done?.citations ?? []);
  const [highlight, setHighlight] = useState<number | null>(null);
  const cardRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const mode = MODES[message.mode];

  const showSource = (n: number) => {
    setHighlight(n);
    cardRefs.current[n]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    window.setTimeout(() => setHighlight((h) => (h === n ? null : h)), 1800);
  };

  const waiting = message.streaming && !message.content;

  return (
    <div className="flex gap-3">
      <div className={cx("mt-1 flex size-7 shrink-0 items-center justify-center rounded-full text-white", mode.accent.bg)}>
        <mode.icon className="size-4" />
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        {message.alert && (
          <div
            role="alert"
            className="flex items-start gap-2.5 rounded-xl border border-red-300 bg-red-50 p-3.5 text-sm font-medium text-red-800 dark:border-red-800 dark:bg-red-950/60 dark:text-red-200"
          >
            <Phone className="mt-0.5 size-4 shrink-0" />
            <span>{message.alert.message}</span>
          </div>
        )}
        {waiting && message.stage && (
          <div className="flex items-center gap-2 py-1.5 text-sm text-zinc-500">
            <Loader2 className="size-4 animate-spin" /> {stageLabel(message.stage, message.mode)}
          </div>
        )}
        {message.content && (
          <Markdown
            text={message.content}
            available={sources.length}
            onCite={showSource}
            streaming={message.streaming && !!message.content}
          />
        )}
        {message.error && (
          <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" /> {message.error}
          </div>
        )}

        {message.done && onAsk && <NextSteps message={message} onAsk={onAsk} />}

        {message.enrichment?.entities?.length ? (
          <div className="flex flex-wrap gap-1.5">
            {message.enrichment.entities.map((e) => (
              <span key={e.text} className={cx("rounded-full px-2 py-0.5 text-xs", mode.accent.soft)}>
                {e.text} <span className="opacity-60">· {e.category}</span>
              </span>
            ))}
          </div>
        ) : null}
        {message.enrichment?.concepts?.length ? (
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-zinc-500">Key concepts:</span>
            {message.enrichment.concepts.map((c) => (
              <span key={c.text} className={cx("rounded-full px-2 py-0.5", mode.accent.soft)}>
                {c.text}
              </span>
            ))}
          </div>
        ) : null}

        {sources.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-xs font-medium tracking-wide text-zinc-500 uppercase">Sources</div>
            <div className="grid gap-1.5 sm:grid-cols-2">
              {sources.map((s) => (
                <SourceCard
                  key={s.id}
                  ref={(el) => {
                    cardRefs.current[s.n] = el;
                  }}
                  source={s}
                  cited={!message.done || cited.has(s.n)}
                  highlighted={highlight === s.n}
                />
              ))}
            </div>
          </div>
        )}

        {message.done && <AnswerFooter message={message} onRegenerate={onRegenerate} onRate={onRate} />}
      </div>
    </div>
  );
}

/** When Billie could not answer: a better-suited mode, and the closest topics this mode does have. */
function NextSteps({ message, onAsk }: { message: UIMessage; onAsk: NonNullable<AssistantProps["onAsk"]> }) {
  const done = message.done!;
  const better = done.suggested_mode ? MODES[done.suggested_mode.mode] : null;
  const related = done.related ?? [];
  if (!better && !related.length) return null;
  return (
    <div className="space-y-2.5 rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-sm dark:border-zinc-800 dark:bg-zinc-900/60">
      {better && (
        <div className="flex flex-wrap items-center gap-2">
          <span>This looks like a question for {better.label} mode.</span>
          <button
            type="button"
            onClick={() => onAsk(undefined, better.id)}
            className={cx("inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium text-white", better.accent.bg)}
          >
            Ask in {better.label} mode <ArrowRight className="size-3.5" />
          </button>
        </div>
      )}
      {related.length > 0 && (
        <div>
          <div className="text-xs text-zinc-500">Closest topics I do have in {MODES[message.mode].label} mode:</div>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {related.map((r) => (
              <button
                key={r.title}
                type="button"
                onClick={() => onAsk(r.question, undefined)}
                className="rounded-full border border-zinc-300 px-2.5 py-1 text-left text-xs hover:bg-white dark:border-zinc-700 dark:hover:bg-zinc-800"
              >
                {r.title}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function AnswerFooter({ message, onRegenerate, onRate }: AssistantProps) {
  const done = message.done!;
  const [showDetails, setShowDetails] = useState(false);
  const [copied, setCopied] = useState(false);
  const timings = done.timings_ms ?? {};
  const analysis = message.analysis ?? done.analysis;

  return (
    <div className="space-y-2 text-xs text-zinc-500">
      {message.enrichment?.disclaimer && (
        <div className="flex items-center gap-1.5 text-amber-700 dark:text-amber-400">
          <ShieldAlert className="size-3.5" /> {message.enrichment.disclaimer}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {done.grounded === true && (
          <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 className="size-3.5" /> Grounded in {done.citations?.length} source
            {done.citations?.length === 1 ? "" : "s"}
          </span>
        )}
        {done.grounded === false && (
          <span className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-400">
            <AlertTriangle className="size-3.5" /> Not fully grounded - check the sources
          </span>
        )}
        {!!done.unverified_sentences && (
          <span className="inline-flex items-center gap-1" title="These sentences could not be matched to any of the sources shown">
            <Info className="size-3.5" /> {done.unverified_sentences} sentence{done.unverified_sentences === 1 ? "" : "s"} not linked to a source
          </span>
        )}
        {done.abstained && (
          <span className="inline-flex items-center gap-1">
            <Info className="size-3.5" /> No sources cited - the documents may not cover this
          </span>
        )}
        {done.not_found && (
          <span className="inline-flex items-center gap-1">
            <Info className="size-3.5" /> No matching documents
          </span>
        )}
        {done.used_llm === false && !done.not_found && (
          <span className="inline-flex items-center gap-1 text-amber-600">
            <Info className="size-3.5" /> Retrieval-only answer
          </span>
        )}
        {done.model && (
          <span className="inline-flex items-center gap-1">
            <Sparkles className="size-3.5" /> {done.model}
          </span>
        )}
        {timings.total_ms !== undefined && <span>{formatMs(timings.total_ms)}</span>}
        {done.cached && (
          <span className="inline-flex items-center gap-1" title="Same question answered earlier with the same model and documents">
            <Zap className="size-3.5" /> cached
          </span>
        )}
        {analysis && analysis.language !== "en" && <span>{analysis.language_name.split(" (")[0]}</span>}
        {analysis?.sentiment === "negative" && <span>tone: empathetic</span>}
        <button
          type="button"
          className="inline-flex items-center gap-1 hover:text-zinc-800 dark:hover:text-zinc-200"
          onClick={() => {
            void navigator.clipboard.writeText(done.answer ?? message.content);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1500);
          }}
        >
          <Copy className="size-3.5" /> {copied ? "Copied" : "Copy"}
        </button>
        {onRate && message.messageId !== undefined && (
          <span className="inline-flex items-center gap-0.5">
            <button
              type="button"
              aria-label="Good answer"
              aria-pressed={message.feedback === 1}
              onClick={() => onRate(1)}
              className={cx("rounded p-1 hover:bg-zinc-100 dark:hover:bg-zinc-800", message.feedback === 1 && "text-emerald-600")}
            >
              <ThumbsUp className={cx("size-3.5", message.feedback === 1 && "fill-current")} />
            </button>
            <button
              type="button"
              aria-label="Bad answer"
              aria-pressed={message.feedback === -1}
              onClick={() => onRate(-1)}
              className={cx("rounded p-1 hover:bg-zinc-100 dark:hover:bg-zinc-800", message.feedback === -1 && "text-red-600")}
            >
              <ThumbsDown className={cx("size-3.5", message.feedback === -1 && "fill-current")} />
            </button>
          </span>
        )}
        {onRegenerate && (
          <button type="button" onClick={onRegenerate} className="inline-flex items-center gap-1 hover:text-zinc-800 dark:hover:text-zinc-200">
            <RefreshCw className="size-3.5" /> Regenerate
          </button>
        )}
        <button
          type="button"
          className="underline-offset-2 hover:text-zinc-800 hover:underline dark:hover:text-zinc-200"
          onClick={() => setShowDetails((v) => !v)}
        >
          {showDetails ? "Hide details" : "Details"}
        </button>
      </div>
      {showDetails && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 rounded-lg bg-zinc-50 p-3 font-mono text-[11px] dark:bg-zinc-900">
          <dt>search</dt>
          <dd>
            {done.retrieval_mode ?? message.retrieval?.mode}
            {message.retrieval?.degraded && ` (degraded: ${message.retrieval.note})`}
          </dd>
          {(message.retrieval?.translated_query ?? done.translated_query) && (
            <>
              <dt>search query</dt>
              <dd>{message.retrieval?.translated_query ?? done.translated_query}</dd>
            </>
          )}
          {analysis && (
            <>
              <dt>language</dt>
              <dd>
                {analysis.language} ({Math.round(analysis.language_confidence * 100)}%, {analysis.method.language})
              </dd>
              <dt>sentiment</dt>
              <dd>
                {analysis.sentiment} ({Math.round(analysis.sentiment_score * 100)}%, {analysis.method.sentiment})
              </dd>
            </>
          )}
          {Object.entries(timings).map(([k, v]) => (
            <FragmentRow key={k} label={k.replace(/_ms$/, "")} value={formatMs(v)} />
          ))}
          {message.imageDescription && (
            <>
              <dt>image</dt>
              <dd className="whitespace-pre-wrap">{message.imageDescription}</dd>
            </>
          )}
          {done.auto_citations?.map((c) => (
            <FragmentRow key={c.sentence} label="auto-cited" value={`[${c.n}] (similarity ${c.score}) ${c.sentence.slice(0, 90)}…`} />
          ))}
          {done.notes?.map((note) => (
            <FragmentRow key={note} label="note" value={note} />
          ))}
        </dl>
      )}
    </div>
  );
}

function FragmentRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}
