import { CheckCircle2, ExternalLink, KeyRound, Loader2, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, loadKeys, saveKeys } from "../api/client";
import type { LLMSettings, ProviderPreset } from "../api/types";
import { Button, Dialog, Field, inputClass } from "./Dialog";

interface Props {
  onClose: () => void;
  onSaved: (settings: LLMSettings) => void;
}

type TestState = { status: "idle" | "testing" } | { status: "done"; ok: boolean; detail: string };

export function SettingsDialog({ onClose, onSaved }: Props) {
  const [presets, setPresets] = useState<ProviderPreset[]>([]);
  const [saved, setSaved] = useState<LLMSettings | null>(null);
  const [provider, setProvider] = useState("ollama");
  const [model, setModel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [visionModel, setVisionModel] = useState("");
  const [temperature, setTemperature] = useState(0.2);
  const [maxTokens, setMaxTokens] = useState(700);
  const [apiKey, setApiKey] = useState("");
  const [installed, setInstalled] = useState<string[]>([]);
  const [test, setTest] = useState<TestState>({ status: "idle" });
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [access, setAccess] = useState(loadKeys());

  useEffect(() => {
    void Promise.all([api.presets(), api.llmSettings()])
      .then(([p, s]) => {
        setPresets(p);
        setSaved(s);
        setProvider(s.provider);
        setModel(s.model);
        setBaseUrl(s.base_url);
        setVisionModel(s.vision_model);
        setTemperature(s.temperature);
        setMaxTokens(s.max_tokens);
      })
      .catch((e: Error) => setError(e.message));
    void api.models().then((r) => setInstalled(r.models)).catch(() => undefined);
  }, []);

  const preset = useMemo(() => presets.find((p) => p.id === provider), [presets, provider]);
  const providerChanged = saved !== null && provider !== saved.provider;
  const storedKey = saved?.keys[provider] ?? null;
  const modelOptions = [...new Set([...(provider === saved?.provider ? installed : []), ...(preset?.suggested_models ?? [])])];

  const pickProvider = (id: string) => {
    setProvider(id);
    const p = presets.find((x) => x.id === id);
    if (p) {
      setModel(id === saved?.provider ? saved.model : p.default_model);
      setBaseUrl(id === saved?.provider ? saved.base_url : p.base_url);
    }
    setApiKey("");
    setTest({ status: "idle" });
  };

  const runTest = async () => {
    setTest({ status: "testing" });
    try {
      const r = await api.testLlm({ provider, model, base_url: baseUrl, api_key: apiKey || undefined });
      setTest({ status: "done", ok: r.ok, detail: r.detail });
    } catch (e) {
      setTest({ status: "done", ok: false, detail: (e as Error).message });
    }
  };

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      saveKeys(access);
      const result = await api.updateLlmSettings({
        provider,
        model,
        base_url: baseUrl,
        vision_model: visionModel,
        temperature,
        max_tokens: maxTokens,
        ...(apiKey ? { api_key: apiKey } : {}),
      });
      onSaved(result);
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog
      title="Model settings"
      description="Choose which language model writes Billie's answers. Search always runs locally."
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={save} disabled={saving || !model}>
            {saving && <Loader2 className="size-4 animate-spin" />} Save
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <Field label="Provider">
          <div className="grid grid-cols-2 gap-1.5">
            {presets.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => pickProvider(p.id)}
                className={`rounded-lg border px-3 py-2 text-left text-sm transition ${
                  provider === p.id
                    ? "border-zinc-900 bg-zinc-900 text-white dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900"
                    : "border-zinc-200 hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </Field>

        {preset?.needs_key && (
          <Field
            label="API key"
            hint={
              <span className="flex flex-wrap items-center gap-x-2">
                {storedKey ? <>Saved key: <code>{storedKey}</code>. Leave empty to keep it.</> : "Stored only on this computer."}
                {preset.key_url && (
                  <a href={preset.key_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-0.5 underline">
                    Get a key <ExternalLink className="size-3" />
                  </a>
                )}
              </span>
            }
          >
            <input
              type="password"
              autoComplete="off"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={storedKey ? "••••••••" : "Paste your API key"}
              className={inputClass}
            />
          </Field>
        )}

        <Field
          label="Model"
          hint={provider === "ollama" && installed.length > 0 ? `${installed.length} models installed in Ollama` : undefined}
        >
          <input list="model-options" value={model} onChange={(e) => setModel(e.target.value)} className={inputClass} />
          <datalist id="model-options">
            {modelOptions.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </Field>

        {(provider === "ollama" || provider === "custom") && (
          <Field label="Server URL">
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} className={inputClass} />
          </Field>
        )}

        {provider === "ollama" && (
          <Field label="Vision model" hint="Used to read attached images when the chat model is text-only.">
            <input list="model-options" value={visionModel} onChange={(e) => setVisionModel(e.target.value)} className={inputClass} />
          </Field>
        )}

        <div className="grid grid-cols-2 gap-3">
          <Field label={`Temperature: ${temperature.toFixed(1)}`} hint="Lower = more factual">
            <input
              type="range"
              min={0}
              max={1}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full"
            />
          </Field>
          <Field label="Max answer tokens">
            <input
              type="number"
              min={64}
              max={8192}
              step={64}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              className={inputClass}
            />
          </Field>
        </div>

        <div className="flex items-center gap-3">
          <Button onClick={runTest} disabled={test.status === "testing" || !model}>
            {test.status === "testing" && <Loader2 className="size-4 animate-spin" />} Test connection
          </Button>
          {test.status === "done" && (
            <span className={`flex items-center gap-1 text-sm ${test.ok ? "text-emerald-600" : "text-red-600"}`}>
              {test.ok ? <CheckCircle2 className="size-4" /> : <XCircle className="size-4" />}
              <span className="line-clamp-2">{test.ok ? "Connected" : test.detail}</span>
            </span>
          )}
        </div>
        {providerChanged && preset?.needs_key && (
          <p className="text-xs text-zinc-500">
            Cloud providers receive your question, the retrieved passages and attached images.
          </p>
        )}

        <details className="rounded-lg border border-zinc-200 p-3 text-sm dark:border-zinc-800">
          <summary className="flex cursor-pointer items-center gap-1.5 font-medium">
            <KeyRound className="size-4" /> Server access keys
          </summary>
          <p className="mt-2 text-xs text-zinc-500">
            Only needed when Billie runs with BILLIE_API_KEY / BILLIE_ADMIN_KEY set. Stored in this browser.
          </p>
          <div className="mt-3 space-y-3">
            <Field label="API key">
              <input
                type="password"
                value={access.apiKey ?? ""}
                onChange={(e) => setAccess({ ...access, apiKey: e.target.value })}
                className={inputClass}
              />
            </Field>
            <Field label="Admin key">
              <input
                type="password"
                value={access.adminKey ?? ""}
                onChange={(e) => setAccess({ ...access, adminKey: e.target.value })}
                className={inputClass}
              />
            </Field>
          </div>
        </details>

        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>
    </Dialog>
  );
}
