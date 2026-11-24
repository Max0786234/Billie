import { Activity, FileText, Moon, PanelLeftClose, Pencil, Plus, Settings, Sun, Trash2 } from "lucide-react";
import { useState } from "react";
import type { ConversationSummary, Mode } from "../api/types";
import { cx, relativeTime } from "../lib/format";
import { MODE_LIST, MODES } from "../lib/modes";

interface Props {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  conversations: ConversationSummary[];
  activeId: string | null;
  onNew: () => void;
  onOpen: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onOpenSettings: () => void;
  onOpenDocuments: () => void;
  onOpenStatus: () => void;
  dark: boolean;
  onToggleDark: () => void;
  onClose: () => void;
}

export function Sidebar(props: Props) {
  const { mode, conversations, activeId } = props;
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  return (
    <aside className="flex h-full w-72 flex-col border-r border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900/60">
      <div className="flex items-center justify-between px-4 pt-4 pb-3">
        <div className="flex items-center gap-2">
          <img src="/favicon.svg" alt="" className="size-7" />
          <span className="text-lg font-semibold tracking-tight">Billie</span>
        </div>
        <button
          type="button"
          onClick={props.onClose}
          className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-200 lg:hidden dark:hover:bg-zinc-800"
          aria-label="Close sidebar"
        >
          <PanelLeftClose className="size-4" />
        </button>
      </div>

      <div className="px-3">
        <button
          type="button"
          onClick={props.onNew}
          className="flex w-full items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm font-medium shadow-sm transition hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-800 dark:hover:bg-zinc-700"
        >
          <Plus className="size-4" /> New chat
        </button>
      </div>

      <nav className="mt-4 space-y-0.5 px-3" aria-label="Mode">
        <div className="px-2 pb-1 text-[11px] font-medium tracking-wide text-zinc-500 uppercase">Mode</div>
        {MODE_LIST.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => props.onModeChange(m.id)}
            aria-pressed={mode === m.id}
            className={cx(
              "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition",
              mode === m.id
                ? "bg-white font-medium shadow-sm dark:bg-zinc-800"
                : "text-zinc-600 hover:bg-zinc-200/60 dark:text-zinc-400 dark:hover:bg-zinc-800/60",
            )}
          >
            <m.icon className={cx("size-4", mode === m.id && m.accent.text)} />
            {m.label}
          </button>
        ))}
      </nav>

      <div className="mt-4 min-h-0 flex-1 overflow-y-auto px-3">
        <div className="px-2 pb-1 text-[11px] font-medium tracking-wide text-zinc-500 uppercase">History</div>
        {conversations.length === 0 && <p className="px-2 py-2 text-xs text-zinc-500">No conversations yet.</p>}
        <ul className="space-y-0.5">
          {conversations.map((c) => {
            const Icon = MODES[c.mode]?.icon ?? MODES.general.icon;
            return (
              <li key={c.id} className="group relative">
                {editing === c.id ? (
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      props.onRename(c.id, draft);
                      setEditing(null);
                    }}
                  >
                    <input
                      autoFocus
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                      onBlur={() => setEditing(null)}
                      className="w-full rounded-lg border border-zinc-300 bg-white px-2.5 py-1.5 text-sm outline-none dark:border-zinc-600 dark:bg-zinc-800"
                    />
                  </form>
                ) : (
                  <button
                    type="button"
                    onClick={() => props.onOpen(c.id)}
                    className={cx(
                      "flex w-full items-start gap-2 rounded-lg px-2.5 py-2 pr-14 text-left text-sm transition",
                      activeId === c.id ? "bg-zinc-200/80 dark:bg-zinc-800" : "hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50",
                    )}
                  >
                    <Icon className={cx("mt-0.5 size-3.5 shrink-0", MODES[c.mode]?.accent.text)} />
                    <span className="min-w-0">
                      <span className="block truncate">{c.title}</span>
                      <span className="text-[11px] text-zinc-500">{relativeTime(c.updated_at)}</span>
                    </span>
                  </button>
                )}
                {editing !== c.id && (
                  <div className="absolute top-1.5 right-1.5 hidden gap-0.5 group-hover:flex">
                    <button
                      type="button"
                      aria-label="Rename"
                      onClick={() => {
                        setDraft(c.title);
                        setEditing(c.id);
                      }}
                      className="rounded p-1 text-zinc-500 hover:bg-zinc-300 hover:text-zinc-800 dark:hover:bg-zinc-700"
                    >
                      <Pencil className="size-3.5" />
                    </button>
                    <button
                      type="button"
                      aria-label="Delete"
                      onClick={() => props.onDelete(c.id)}
                      className="rounded p-1 text-zinc-500 hover:bg-red-100 hover:text-red-600 dark:hover:bg-red-950"
                    >
                      <Trash2 className="size-3.5" />
                    </button>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </div>

      <div className="space-y-0.5 border-t border-zinc-200 p-3 dark:border-zinc-800">
        <SidebarButton icon={FileText} label="Documents" onClick={props.onOpenDocuments} />
        <SidebarButton icon={Activity} label="System status" onClick={props.onOpenStatus} />
        <SidebarButton icon={Settings} label="Model settings" onClick={props.onOpenSettings} />
        <SidebarButton icon={props.dark ? Sun : Moon} label={props.dark ? "Light theme" : "Dark theme"} onClick={props.onToggleDark} />
      </div>
    </aside>
  );
}

function SidebarButton({ icon: Icon, label, onClick }: { icon: typeof Plus; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-zinc-600 transition hover:bg-zinc-200/60 dark:text-zinc-400 dark:hover:bg-zinc-800/60"
    >
      <Icon className="size-4" /> {label}
    </button>
  );
}
