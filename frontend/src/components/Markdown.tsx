import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cx } from "../lib/format";

const CITATION = /\[(\d{1,2})\](?!\()/g;

/** Turn "[2]" into a link the renderer recognises, without touching real Markdown links. */
export function linkCitations(text: string): string {
  return text.replace(CITATION, (_, n: string) => `[${n}](#cite-${n})`);
}

interface Props {
  text: string;
  available: number;
  onCite?: (n: number) => void;
  streaming?: boolean;
}

export const Markdown = memo(function Markdown({ text, available, onCite, streaming }: Props) {
  return (
    <div
      className={cx(
        "prose prose-zinc max-w-none dark:prose-invert prose-p:my-2 prose-ul:my-2 prose-li:my-0.5 prose-headings:mt-4",
        streaming && "caret",
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const match = href?.match(/^#cite-(\d+)$/);
            if (match) {
              const n = Number(match[1]);
              const valid = n >= 1 && n <= available;
              return (
                <button
                  type="button"
                  onClick={() => valid && onCite?.(n)}
                  title={valid ? `Show source ${n}` : "This citation does not match any source"}
                  className={cx(
                    "mx-0.5 inline-flex h-5 min-w-5 -translate-y-px items-center justify-center rounded px-1 align-middle text-[11px] font-semibold no-underline transition",
                    valid
                      ? "bg-zinc-200 text-zinc-700 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-600"
                      : "bg-red-100 text-red-700 line-through dark:bg-red-950 dark:text-red-300",
                  )}
                >
                  {children}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer noopener">
                {children}
              </a>
            );
          },
        }}
      >
        {linkCitations(text)}
      </ReactMarkdown>
    </div>
  );
});
