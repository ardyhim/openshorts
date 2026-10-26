import React, { useRef, useState } from 'react';
import { AlertCircle, Check, Download, FileJson, Loader2, Upload } from 'lucide-react';
import { apiFetch } from '../lib/api';

const EXAMPLE = `{
  "schema": "openshorts.clip-selection.v1",
  "clips": [
    {
      "start": 12.5,
      "end": 42.0,
      "score": 85,
      "title": "Your YouTube Shorts title",
      "hook": "The strongest 10-word hook",
      "reason": "Why this moment works",
      "description_tiktok": "Teaser + hashtags",
      "description_instagram": "Teaser + hashtags"
    }
  ]
}`;

function formatDuration(seconds) {
    const total = Math.max(0, Math.round(Number(seconds) || 0));
    return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

function errorText(payload, fallback) {
    const detail = payload?.detail;
    if (typeof detail === 'string' && detail) return detail;
    if (detail && typeof detail.message === 'string') return detail.message;
    return fallback;
}

export default function ManualAnalysisPanel({ jobId, info, variant = 'manual', onImported }) {
    const [busy, setBusy] = useState('');
    const [error, setError] = useState('');
    const [warnings, setWarnings] = useState([]);
    const [imported, setImported] = useState(false);
    const fileRef = useRef(null);
    const unavailable = info?.available === false;

    const downloadTranscript = async () => {
        setError('');
        setBusy('download');
        try {
            const res = await apiFetch(`/api/jobs/${jobId}/analysis/export`);
            if (!res.ok) {
                const payload = await res.json().catch(() => ({}));
                throw new Error(errorText(payload, 'Could not export the transcript.'));
            }
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `openshorts_transcript_${String(jobId).slice(0, 8)}.json`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        } catch (e) {
            setError(e.message || 'Could not export the transcript.');
        } finally {
            setBusy('');
        }
    };

    const importFile = async (event) => {
        const file = event.target.files?.[0];
        event.target.value = '';
        if (!file) return;
        if (file.size > 5 * 1024 * 1024) {
            setError('That file is larger than 5 MB. The clip selection should be tiny.');
            return;
        }
        setError('');
        setWarnings([]);
        setImported(false);
        setBusy('import');
        try {
            const text = await file.text();
            const res = await apiFetch(`/api/jobs/${jobId}/analysis/import`, {
                method: 'POST',
                headers: { 'Content-Type': 'text/plain; charset=utf-8' },
                body: text,
            });
            const payload = await res.json().catch(() => ({}));
            if (!res.ok) {
                throw new Error(errorText(payload, 'The clip selection could not be imported.'));
            }
            setWarnings(payload.warnings || []);
            setImported(true);
            if (onImported) onImported(payload);
        } catch (e) {
            setError(e.message || 'The clip selection could not be imported.');
        } finally {
            setBusy('');
        }
    };

    return (
        <div className="rounded-card border border-brass/40 bg-brass/5 p-4 sm:p-5 space-y-4">
            <div className="flex items-start gap-3">
                <div className="shrink-0 w-9 h-9 rounded-full bg-brass/15 text-brass flex items-center justify-center">
                    <FileJson size={18} />
                </div>
                <div className="min-w-0">
                    <h3 className="text-sm font-medium text-ink lowercase">
                        {variant === 'failed' ? 'get clips without gemini' : 'manual ai selection'}
                    </h3>
                    <p className="mt-1 text-xs sm:text-[13px] text-ink2 leading-relaxed">
                        {variant === 'failed'
                            ? 'This job stopped before the clips were ready, but the transcript is safe. Download it, let any AI pick the hook moments, then import the JSON — no Gemini calls are needed for rendering.'
                            : 'The transcript is ready and Gemini is not used for this job. Download the JSON, ask any AI for the best hook timestamps, then import its answer.'}
                    </p>
                </div>
            </div>

            {info?.available && (
                <div className="flex flex-wrap gap-2 text-[11px] readout text-muted">
                    <span className="bg-paper3 px-2 py-1 rounded-full">
                        {formatDuration(info.duration_seconds)}
                    </span>
                    <span className="bg-paper3 px-2 py-1 rounded-full">
                        {info.segments} segments
                    </span>
                    <span className="bg-paper3 px-2 py-1 rounded-full">
                        {info.windows} windows
                    </span>
                    {info.language && (
                        <span className="bg-paper3 px-2 py-1 rounded-full">{info.language}</span>
                    )}
                    {info.recommended_clip_count && (
                        <span className="bg-paper3 px-2 py-1 rounded-full">
                            aim {info.recommended_clip_count.min}-{info.recommended_clip_count.max} clips
                        </span>
                    )}
                </div>
            )}

            <ol className="text-xs sm:text-[13px] text-ink2 space-y-1.5 list-decimal list-inside leading-relaxed">
                <li><b className="text-ink">Download</b> the transcript JSON. It contains the timed transcript, candidate windows and a ready-made prompt.</li>
                <li>Give the file to ChatGPT, Claude, Gemini web, a local model, or any AI you like. It should return one JSON object.</li>
                <li><b className="text-ink">Import</b> that reply here. The app validates it, snaps cuts to word boundaries and renders the clips.</li>
            </ol>

            {unavailable && (
                <p className="flex items-start gap-2 text-xs text-danger leading-relaxed">
                    <AlertCircle size={14} className="shrink-0 mt-0.5" />
                    This job has no transcript to export. Manual analysis needs a source with speech.
                </p>
            )}

            {error && (
                <p className="flex items-start gap-2 text-xs text-danger leading-relaxed">
                    <AlertCircle size={14} className="shrink-0 mt-0.5" />
                    {error}
                </p>
            )}

            {warnings.length > 0 && (
                <div className="text-xs text-muted space-y-1">
                    {warnings.map((warning, i) => (
                        <p key={i} className="flex items-start gap-2">
                            <AlertCircle size={12} className="shrink-0 mt-0.5" />
                            {warning}
                        </p>
                    ))}
                </div>
            )}

            <div className="flex flex-col sm:flex-row gap-2">
                <button
                    type="button"
                    onClick={downloadTranscript}
                    disabled={!!busy || unavailable}
                    className="btn-primary px-3 py-2 text-xs justify-center"
                >
                    {busy === 'download'
                        ? <><Loader2 size={14} className="animate-spin" /> preparing…</>
                        : <><Download size={14} /> download transcript</>}
                </button>
                <button
                    type="button"
                    onClick={() => fileRef.current?.click()}
                    disabled={!!busy || unavailable}
                    className="btn-ghost px-3 py-2 text-xs justify-center"
                >
                    {busy === 'import'
                        ? <><Loader2 size={14} className="animate-spin" /> importing…</>
                        : <><Upload size={14} /> import ai result</>}
                </button>
                <input
                    ref={fileRef}
                    type="file"
                    accept=".json,.txt,application/json,text/plain"
                    onChange={importFile}
                    className="hidden"
                />
            </div>

            <details className="text-xs text-muted">
                <summary className="cursor-pointer hover:text-ink2 select-none">
                    expected JSON shape
                </summary>
                <pre className="mt-2 p-3 rounded-input bg-paper border border-rule overflow-x-auto text-[11px] leading-relaxed">
                    {EXAMPLE}
                </pre>
                <p className="mt-2 leading-relaxed">
                    Markdown fences are fine. The app also accepts the internal
                    <code className="mx-1 text-ink2">shorts</code>
                    field names, so the AI may copy the app&apos;s own metadata format.
                </p>
            </details>

            {imported && !busy && (
                <p className="flex items-center gap-2 text-xs text-ok">
                    <Check size={14} /> Imported. Rendering will start in the background.
                </p>
            )}
        </div>
    );
}
