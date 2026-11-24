import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { SystemStatus } from "../api/types";
import { Button, Dialog } from "./Dialog";

const CORPUS_LABEL = { knowledge: "Knowledge base", medquad: "MedQuAD (medical)", arxiv: "arXiv (research)" };

export function StatusDialog({ onClose }: { onClose: () => void }) {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.status().then(setStatus).catch((e: Error) => setError(e.message));
  }, []);

  return (
    <Dialog title="System status" onClose={onClose} footer={<Button onClick={onClose}>Close</Button>}>
      {!status && !error && <Loader2 className="mx-auto size-5 animate-spin text-zinc-400" />}
      {error && <p className="text-sm text-red-600">{error}</p>}
      {status && (
        <div className="space-y-5 text-sm">
          <section className="space-y-2">
            <h3 className="font-medium">Language model</h3>
            <StatusRow ok={status.llm.ok} label={`${status.llm.provider} · ${status.llm.model}`} detail={status.llm.ok ? "ready" : status.llm.detail} />
          </section>
          <section className="space-y-2">
            <h3 className="font-medium">Semantic search</h3>
            <StatusRow
              ok={status.embedding.available}
              label={status.embedding.model ?? "disabled"}
              detail={status.embedding.available ? "embedding model ready" : "keyword (BM25) search only"}
            />
          </section>
          <section className="grid grid-cols-2 gap-3">
            <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-800/60">
              <div className="text-xs text-zinc-500">Answer cache</div>
              <div className="mt-0.5 font-medium tabular-nums">
                {status.cache ? `${status.cache.entries} answers · ${status.cache.hits} hits` : "off"}
              </div>
            </div>
            <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-800/60">
              <div className="text-xs text-zinc-500">Feedback</div>
              <div className="mt-0.5 font-medium tabular-nums">
                {(() => {
                  const totals = Object.values(status.feedback ?? {}).reduce(
                    (acc, f) => ({ up: acc.up + f.up, down: acc.down + f.down }),
                    { up: 0, down: 0 },
                  );
                  return `👍 ${totals.up} · 👎 ${totals.down}`;
                })()}
              </div>
            </div>
          </section>
          <section className="space-y-2">
            <h3 className="font-medium">Indexes {status.indexing && <span className="text-xs font-normal text-zinc-500">(updating…)</span>}</h3>
            <table className="w-full text-left">
              <thead className="text-xs text-zinc-500">
                <tr>
                  <th className="pb-1 font-normal">Corpus</th>
                  <th className="pb-1 text-right font-normal">Passages</th>
                  <th className="pb-1 text-right font-normal">Embedded</th>
                  <th className="pb-1 text-right font-normal">Search</th>
                </tr>
              </thead>
              <tbody>
                {status.corpora.map((c) => (
                  <tr key={c.name} className="border-t border-zinc-100 dark:border-zinc-800">
                    <td className="py-1.5">
                      {CORPUS_LABEL[c.name]}
                      {c.variant === "sample" && <span className="ml-1.5 rounded bg-amber-100 px-1 text-[10px] text-amber-800">sample</span>}
                    </td>
                    <td className="py-1.5 text-right tabular-nums">{c.passages.toLocaleString()}</td>
                    <td className="py-1.5 text-right tabular-nums">{c.vectors.toLocaleString()}</td>
                    <td className="py-1.5 text-right">{c.search_mode}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      )}
    </Dialog>
  );
}

function StatusRow({ ok, label, detail }: { ok: boolean; label: string; detail: string }) {
  return (
    <div className="flex items-start gap-2">
      {ok ? <CheckCircle2 className="mt-0.5 size-4 text-emerald-600" /> : <XCircle className="mt-0.5 size-4 text-red-500" />}
      <div>
        <div className="font-mono text-xs">{label}</div>
        <div className="text-xs text-zinc-500">{detail}</div>
      </div>
    </div>
  );
}
