import { ChevronDown, ExternalLink } from "lucide-react";
import { forwardRef, useState } from "react";
import type { Source } from "../api/types";
import { cx } from "../lib/format";

interface Props {
  source: Source;
  cited: boolean;
  highlighted: boolean;
}

export const SourceCard = forwardRef<HTMLDivElement, Props>(function SourceCard({ source, cited, highlighted }, ref) {
  const [open, setOpen] = useState(false);
  const meta = source.metadata as Record<string, string | number | undefined>;
  const subtitle = [source.source, meta.year, meta.primary_category].filter(Boolean).join(" · ");
  return (
    <div
      ref={ref}
      className={cx(
        "rounded-lg border bg-white text-sm transition dark:bg-zinc-900",
        highlighted
          ? "border-zinc-400 ring-2 ring-zinc-400/40 dark:border-zinc-500"
          : "border-zinc-200 dark:border-zinc-800",
        !cited && "opacity-70",
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-2.5 p-2.5 text-left"
        aria-expanded={open}
      >
        <span
          className={cx(
            "mt-0.5 flex h-5 min-w-5 items-center justify-center rounded text-[11px] font-semibold",
            cited ? "bg-zinc-800 text-white dark:bg-zinc-200 dark:text-zinc-900" : "bg-zinc-100 text-zinc-500 dark:bg-zinc-800",
          )}
        >
          {source.n}
        </span>
        <span className="min-w-0 flex-1">
          <span className="line-clamp-2 font-medium text-zinc-800 dark:text-zinc-100">{source.title}</span>
          <span className="mt-0.5 block truncate text-xs text-zinc-500">{subtitle}</span>
        </span>
        <ChevronDown className={cx("mt-0.5 size-4 shrink-0 text-zinc-400 transition", open && "rotate-180")} />
      </button>
      {open && (
        <div className="space-y-2 border-t border-zinc-100 px-3 py-2.5 dark:border-zinc-800">
          {typeof meta.authors === "string" && meta.authors && (
            <p className="line-clamp-2 text-xs text-zinc-500">{meta.authors}</p>
          )}
          <p className="whitespace-pre-line text-[13px] leading-relaxed text-zinc-700 dark:text-zinc-300">{source.text}</p>
          <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px] text-zinc-500">
            {Object.entries(source.components).map(([name, rank]) => (
              <span key={name} className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono dark:bg-zinc-800">
                {name} #{rank}
              </span>
            ))}
            {source.url && (
              <a
                href={source.url}
                target="_blank"
                rel="noreferrer noopener"
                className="ml-auto inline-flex items-center gap-1 font-medium text-zinc-600 hover:text-zinc-900 dark:text-zinc-300 dark:hover:text-white"
              >
                Open original <ExternalLink className="size-3" />
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
});
