import { FileText, Loader2, ShieldAlert, Trash2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { DocumentInfo, PrivateDocuments } from "../api/types";
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

/**
 * Public demo: the visitor's own documents. Private to this browser, searched in General mode,
 * deleted automatically after an hour without use.
 */
export function PrivateDocumentsDialog({ onClose, onAdmin }: { onClose: () => void; onAdmin: () => void }) {
  const [state, setState] = useState<PrivateDocuments | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.privateDocuments().then(setState).catch((e: Error) => setError(e.message));
  }, []);

  const run = async (label: string, action: () => Promise<PrivateDocuments>) => {
    setBusy(label);
    setError(null);
    try {
      setState(await action());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const upload = async (files: FileList) => {
    for (const file of files) await run(`Reading and indexing ${file.name}…`, () => api.uploadPrivateDocument(file));
  };

  const limits = state?.limits;
  const docs = state?.documents ?? [];
  const full = limits ? docs.length >= limits.max_files : false;
  const expires = state?.expires_at ? new Date(state.expires_at * 1000) : null;

  return (
    <Dialog
      title="Your private documents"
      description={`Ask questions about your own files in General mode. Only you can search them, and they are deleted ${limits?.ttl_minutes ?? 60} minutes after you last use them.`}
      onClose={onClose}
      footer={
        <>
          <button type="button" onClick={onAdmin} className="mr-auto text-xs text-zinc-400 hover:text-zinc-600">
            Admin: shared documents
          </button>
          <Button onClick={onClose}>Done</Button>
        </>
      }
    >
      <div className="mb-3 flex items-start gap-2 rounded-lg bg-amber-50 p-2.5 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
        <ShieldAlert className="mt-0.5 size-3.5 shrink-0" />
        Don't upload sensitive or personal documents: passages from them are sent to the AI model to write answers.
      </div>
      <input
        ref={fileRef}
        type="file"
        accept=".pdf,.md,.txt"
        hidden
        onChange={(e) => {
          if (e.target.files?.length) void upload(e.target.files);
          e.target.value = "";
        }}
      />
      <button
        type="button"
        disabled={!!busy || full}
        onClick={() => fileRef.current?.click()}
        className="flex w-full flex-col items-center gap-2 rounded-xl border-2 border-dashed border-zinc-200 px-4 py-6 text-sm text-zinc-500 transition hover:border-zinc-400 hover:text-zinc-700 disabled:opacity-60 dark:border-zinc-700 dark:hover:border-zinc-500"
      >
        {busy ? <Loader2 className="size-6 animate-spin" /> : <Upload className="size-6" />}
        {busy ??
          (full
            ? `Limit reached: ${limits?.max_files} documents. Remove one to add another.`
            : `Upload a PDF, Markdown or text file (up to ${limits?.max_mb ?? 5} MB, about 25 pages of text)`)}
      </button>
      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      <ul className="mt-4 divide-y divide-zinc-100 dark:divide-zinc-800">
        {docs.length === 0 && <li className="py-3 text-sm text-zinc-500">No documents yet.</li>}
        {docs.map((d) => (
          <li key={d.name} className="flex items-center gap-3 py-2.5 text-sm">
            <FileText className="size-4 shrink-0 text-zinc-400" />
            <span className="min-w-0 flex-1">
              <span className="block truncate font-medium">{d.name}</span>
              <span className="text-xs text-zinc-500">
                {formatBytes(d.size)} · {d.passages} passage{d.passages === 1 ? "" : "s"} searchable
              </span>
            </span>
            <button
              type="button"
              aria-label={`Remove ${d.name}`}
              disabled={!!busy}
              onClick={() => void run(`Removing ${d.name}…`, () => api.deletePrivateDocument(d.name))}
              className="rounded-md p-1.5 text-zinc-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950"
            >
              <Trash2 className="size-4" />
            </button>
          </li>
        ))}
      </ul>
      {expires && docs.length > 0 && (
        <p className="mt-3 text-xs text-zinc-500">
          Kept until about {expires.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} unless you keep using
          them. Ask in General mode to search them.
        </p>
      )}
    </Dialog>
  );
}
