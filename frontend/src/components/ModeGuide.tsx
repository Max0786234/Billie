import { Check, FileUp, Sparkles, X } from "lucide-react";
import type { Mode, ModeOverview } from "../api/types";
import { cx } from "../lib/format";
import { MODES } from "../lib/modes";

const QUESTION_TYPES: Record<string, string> = {
  information: "What is …",
  symptoms: "Symptoms",
  treatment: "Treatment",
  inheritance: "Inheritance",
  frequency: "How common",
  "genetic changes": "Genetic causes",
  causes: "Causes",
  "exams and tests": "Tests & diagnosis",
  research: "Research",
  outlook: "Outlook",
  susceptibility: "Who is at risk",
  considerations: "Special considerations",
  prevention: "Prevention",
  stages: "Stages",
  complications: "Complications",
};

const fmt = (n: number) => n.toLocaleString("en-US");

/** One line saying how much is inside, computed from the live index. */
function summary(mode: Mode, o: ModeOverview): string {
  if (mode === "medical") {
    const q = o.questions ?? o.vectors;
    return `${fmt(q)} expert answers on ${o.topics ? fmt(o.topics) : "thousands of"} health topics from U.S. National Institutes of Health sources.`;
  }
  if (mode === "research") {
    const years = o.years ? ` (${o.years[0]}–${o.years[1]})` : "";
    const how =
      o.vectors && o.vectors < o.passages
        ? `searched by keywords; the newest ${fmt(o.vectors)} also by meaning`
        : "searched by keywords and by meaning";
    return `${fmt(o.passages)} computer-science papers from arXiv${years}, ${how}.`;
  }
  const docs = o.documents?.length ?? 0;
  return `Billie's own help pages${docs ? ` (${docs} documents)` : ""} plus any documents you upload.`;
}

interface Props {
  mode: Mode;
  overview?: ModeOverview;
  onPick: (question: string) => void;
  /** General mode: open the documents dialog; ``privateUploads`` = public demo (temporary, private files). */
  onOpenDocuments?: () => void;
  privateUploads?: boolean;
}

/** Start screen: what this mode contains, what it is good for, what it can't answer, and examples that work. */
export function ModeGuide({ mode, overview, onPick, onOpenDocuments, privateUploads }: Props) {
  const info = MODES[mode];
  const chips =
    mode === "medical"
      ? (overview?.question_types ?? []).slice(0, 10).map((t) => QUESTION_TYPES[t.name] ?? t.name)
      : mode === "research"
        ? (overview?.categories ?? []).slice(0, 8).map((c) => c.name)
        : (overview?.documents ?? []).map((d) => d.name.replace(/\.(md|txt|pdf)$/i, "").replaceAll("_", " "));
  const chipLabel = mode === "medical" ? "Kinds of questions" : mode === "research" ? "Biggest areas" : "Documents";
  const examples = overview?.examples?.length ? overview.examples : info.examples;

  return (
    <div className="flex flex-col items-center pt-[6vh] text-center">
      <div className={cx("flex size-12 items-center justify-center rounded-2xl text-white shadow-sm", info.accent.bg)}>
        <info.icon className="size-6" />
      </div>
      <h2 className="mt-4 text-2xl font-semibold tracking-tight">{info.label} mode</h2>
      <p className="mt-1.5 max-w-lg text-sm text-zinc-500">
        Billie answers only from what is inside this mode, and cites it. Here is what that is.
      </p>

      <section
        aria-label={`What ${info.label} mode contains`}
        className="mt-6 w-full max-w-2xl rounded-2xl border border-zinc-200 p-4 text-left text-sm dark:border-zinc-800"
      >
        <p className="font-medium">{overview ? summary(mode, overview) : info.tagline}</p>
        {mode === "medical" && overview?.sources?.length ? (
          <p className="mt-1 text-xs text-zinc-500">
            Sources: {overview.sources.slice(0, 6).map((s) => s.name).join(" · ")}
          </p>
        ) : null}

        {chips.length > 0 && (
          <div className="mt-3">
            <div className="text-[11px] font-medium tracking-wide text-zinc-500 uppercase">{chipLabel}</div>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {chips.map((c) => (
                <span key={c} className={cx("rounded-full px-2 py-0.5 text-xs", info.accent.soft)}>
                  {c}
                </span>
              ))}
            </div>
          </div>
        )}

        {mode === "general" && onOpenDocuments && (
          <button
            type="button"
            onClick={onOpenDocuments}
            className={cx("mt-3 flex w-full items-start gap-2 rounded-xl px-3 py-2 text-left text-xs", info.accent.soft)}
          >
            <FileUp className="mt-0.5 size-4 shrink-0" />
            <span>
              <b>Have your own document?</b>{" "}
              {privateUploads
                ? "Upload a PDF, Markdown or text file and ask about it here. Only you can search it, and it is deleted an hour after you last use it."
                : "Upload a PDF, Markdown or text file and General mode will search it too."}
            </span>
          </button>
        )}

        {overview && (
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <div>
              <div className="text-[11px] font-medium tracking-wide text-zinc-500 uppercase">Good for</div>
              <ul className="mt-1 space-y-1">
                {overview.good_for.map((g) => (
                  <li key={g} className="flex items-start gap-1.5">
                    <Check className="mt-0.5 size-3.5 shrink-0 text-emerald-600" /> {g}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <div className="text-[11px] font-medium tracking-wide text-zinc-500 uppercase">Not covered</div>
              <ul className="mt-1 space-y-1 text-zinc-600 dark:text-zinc-400">
                {overview.not_covered.map((g) => (
                  <li key={g} className="flex items-start gap-1.5">
                    <X className="mt-0.5 size-3.5 shrink-0 text-zinc-400" /> {g}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </section>

      <div className="mt-5 flex items-center gap-1.5 text-xs font-medium tracking-wide text-zinc-500 uppercase">
        <Sparkles className="size-3.5" /> Try one of these
      </div>
      <div className="mt-2 grid w-full max-w-2xl gap-2 sm:grid-cols-2">
        {examples.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="rounded-xl border border-zinc-200 px-4 py-2.5 text-left text-sm text-zinc-700 transition hover:bg-zinc-50 dark:border-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-900"
          >
            {q}
          </button>
        ))}
      </div>
      <p className="mt-4 text-xs text-zinc-400">Start typing below to see matching questions from the index.</p>
    </div>
  );
}
