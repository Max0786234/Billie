import { FileText, Loader2, Trash2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { DocumentInfo } from "../api/types";
import { formatBytes, relativeTime } from "../lib/format";
import { Button, Dialog } from "./Dialog";

export function DocumentsDialog({ onClose }: { onClose: () => void }) {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = () => void api.documents().then(setDocs).catch((e: Error) => setError(e.message));
  useEffect(refresh, []);

  const upload = async (files: FileList) => {
    setBusy(true);
    setError(null);
    try {
      for (const file of files) await api.uploadDocument(file);
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (name: string) => {
    setBusy(true);
    try {
      await api.deleteDocument(name);
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      title="Your documents"
      description="Files added here become searchable in General mode (PDF, Markdown or text, up to 20 MB)."
      onClose={onClose}
      footer={<Button onClick={onClose}>Done</Button>}
    >
      <input
        ref={fileRef}
        type="file"
        accept=".pdf,.md,.txt"
        multiple
        hidden
        onChange={(e) => {
          if (e.target.files?.length) void upload(e.target.files);
          e.target.value = "";
        }}
      />
      <button
        type="button"
        disabled={busy}
        onClick={() => fileRef.current?.click()}
        className="flex w-full flex-col items-center gap-2 rounded-xl border-2 border-dashed border-zinc-200 px-4 py-6 text-sm text-zinc-500 transition hover:border-zinc-400 hover:text-zinc-700 dark:border-zinc-700 dark:hover:border-zinc-500"
      >
        {busy ? <Loader2 className="size-6 animate-spin" /> : <Upload className="size-6" />}
        {busy ? "Indexing…" : "Click to upload documents"}
      </button>
      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      <ul className="mt-4 divide-y divide-zinc-100 dark:divide-zinc-800">
        {docs.length === 0 && <li className="py-3 text-sm text-zinc-500">No documents uploaded yet.</li>}
        {docs.map((d) => (
          <li key={d.name} className="flex items-center gap-3 py-2.5 text-sm">
            <FileText className="size-4 shrink-0 text-zinc-400" />
            <span className="min-w-0 flex-1">
              <span className="block truncate font-medium">{d.name}</span>
              <span className="text-xs text-zinc-500">
                {formatBytes(d.size)} · {relativeTime(d.modified)}
              </span>
            </span>
            <button
              type="button"
              aria-label={`Delete ${d.name}`}
              disabled={busy}
              onClick={() => void remove(d.name)}
              className="rounded-md p-1.5 text-zinc-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950"
            >
              <Trash2 className="size-4" />
            </button>
          </li>
        ))}
      </ul>
    </Dialog>
  );
}
