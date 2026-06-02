'use client';

import React, { useRef, useState, useCallback } from 'react';
import { useEmotion } from '@/lib/EmotionContex';

// Запросы идут через Next.js proxy (/api/* → http://localhost:8000/api/*)
// без CORS. Не используем прямой URL бекенда из браузера.

const EMOTION_COLORS: Record<string, string> = {
  happy:     '#FFF18A',
  happiness: '#FFF18A',
  sad:       '#96A4FF',
  sadness:   '#96A4FF',
  anger:     '#FF8D8D',
  surprise:  '#FFC080',
  disgust:   '#B7FF9C',
  fear:      '#BE91FC',
};

export default function AudioSection() {
  const { addResult } = useEmotion();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortRef     = useRef<AbortController | null>(null);

  const [dragging,        setDragging]        = useState(false);
  const [fileName,        setFileName]        = useState<string | null>(null);
  const [file,            setFile]            = useState<File | null>(null);
  const [result,          setResult]          = useState<any | null>(null);
  const [error,           setError]           = useState<string | null>(null);
  const [localProcessing, setLocalProcessing] = useState(false);

  const handleFile = (f: File) => { setFile(f); setFileName(f.name); setResult(null); setError(null); };
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0]; if (f) handleFile(f);
  }, []);

  const analyze = async () => {
    if (!file || localProcessing) return;
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLocalProcessing(true); setError(null); setResult(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`/api/analyze/audio`, { method: 'POST', body: form, signal: controller.signal });
      if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail ?? `HTTP ${res.status}`); }
      const data = await res.json();
      setResult(data);
      addResult({
        emotion: data.emotion, confidence: data.confidence,
        allEmotions: Object.entries(data.probabilities as Record<string, number>).map(([label, score]) => ({ label, score })),
        source: 'audio', timestamp: new Date(),
      });
    } catch (e: any) {
      if (e.name === 'AbortError') return;
      setError(e.message ?? 'Unknown error');
    } finally { setLocalProcessing(false); abortRef.current = null; }
  };

  const topEmotion = result
    ? Object.entries(result.probabilities as Record<string, number>).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <section
      className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-x-16 px-12 py-20 max-md:px-5 max-md:py-12 border-t border-[#e8e8e8] items-start"
      id="audio"
    >
      {/* ── Col 1: header ── */}
      <div>
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold tracking-[0.08em] uppercase text-[#111] opacity-50 mb-4">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-current" />
          Audio Analysis
        </span>
        <h2 className="font-display font-black tracking-[-0.03em] leading-[1.1] mb-6" style={{ fontSize: 'clamp(28px, 4vw, 48px)' }}>
          Upload<br /><span className="italic font-normal">Audio File</span>
        </h2>
        <p className="text-base lg:text-lg text-[#666] leading-[1.75] mt-3">
          Supports WAV, MP3, M4A, FLAC, OGG. The model extracts COVAREP acoustic
          features via opensmile and transcribes speech with Whisper for joint
          audio + text inference.
        </p>
      </div>

      {/* ── Col 2: dropzone + button ── */}
      <div className="flex flex-col gap-3">
        <div
          className={`flex flex-col items-center justify-center gap-2 border-[1.5px] rounded-xl px-6 py-9 cursor-pointer text-center select-none transition-all duration-200 ${
            dragging ? 'border-red-400 bg-red-500/[0.06]'
            : fileName ? 'border-solid border-[#111] opacity-85'
            : 'border-dashed border-[#d4d4d4] hover:border-red-400 hover:bg-red-500/[0.06]'
          }`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input ref={fileInputRef} type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.aac,.opus,.wma" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
          <div className="text-[#111] opacity-35 mb-1">
            <svg width="36" height="36" viewBox="0 0 32 32" fill="none">
              <rect x="4" y="20" width="3" height="8" rx="1.5" fill="currentColor" opacity=".4"/>
              <rect x="9" y="14" width="3" height="14" rx="1.5" fill="currentColor" opacity=".6"/>
              <rect x="14" y="8" width="3" height="20" rx="1.5" fill="currentColor"/>
              <rect x="19" y="12" width="3" height="16" rx="1.5" fill="currentColor" opacity=".7"/>
              <rect x="24" y="17" width="3" height="11" rx="1.5" fill="currentColor" opacity=".4"/>
            </svg>
          </div>
          {fileName
            ? <p className="text-base font-medium text-[#111] opacity-70 break-all">{fileName}</p>
            : <><p className="text-base font-medium text-[#111]">Drop audio file here</p><p className="text-sm text-[#111] opacity-40">or click to browse</p></>
          }
        </div>

        <button
          className="btn-press inline-flex items-center justify-center w-full py-3.5 px-6 text-base font-semibold rounded-lg border-none bg-[#111] text-white cursor-pointer hover:opacity-80 transition-opacity duration-150 disabled:opacity-30 disabled:cursor-not-allowed"
          onClick={analyze} disabled={!file || localProcessing}
        >
          {localProcessing ? 'Analysing…' : 'Analyse Audio →'}
        </button>

        {error && (
          <div className="flex items-start gap-2 p-3.5 bg-[#fff1f1] border border-[#fcc] rounded-lg text-base text-[#c00]">
            <span>⚠</span> {error}
          </div>
        )}
      </div>

      {/* ── Full-width result panel ── */}
      {result && (
        <div className="md:col-span-2 border-[1.5px] border-[#e4e4e4] rounded-2xl overflow-hidden mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2">

            {/* Left: dominant emotion */}
            <div
              className="flex flex-col items-center justify-center gap-4 p-10 text-center"
              style={{ background: EMOTION_COLORS[result.emotion] ?? '#f0f0f0' }}
            >
              <span className="text-xs font-bold tracking-[0.15em] uppercase opacity-50">
                Detected emotion
              </span>
              <span
                className="font-goldman font-black capitalize leading-none"
                style={{ fontSize: 'clamp(40px, 6vw, 80px)' }}
              >
                {result.emotion}
              </span>
              <span
                className="font-goldman font-black leading-none tabular-nums"
                style={{ fontSize: 'clamp(32px, 4vw, 60px)' }}
              >
                {result.confidence.toFixed(2)}%
              </span>
              {result.transcript && (
                <p className="text-sm opacity-60 italic leading-[1.5] max-w-[260px]">
                  "{result.transcript}"
                </p>
              )}
              <span className="text-xs opacity-35 mt-1">{result.latency_ms} ms</span>
            </div>

            {/* Right: all emotions */}
            <div className="p-8 flex flex-col gap-4 border-t md:border-t-0 md:border-l border-[#e4e4e4] bg-white">
              <p className="text-sm font-bold tracking-[0.1em] uppercase text-[#111] opacity-40">
                All emotions
              </p>
              <div className="flex flex-col gap-4">
                {topEmotion.map(([label, score]) => {
                  const isTop = label === result.emotion;
                  return (
                    <div key={label} className="flex items-center gap-3">
                      <span className={`text-lg capitalize text-[#111] w-24 shrink-0 ${isTop ? 'font-bold' : 'font-medium opacity-70'}`}>
                        {label}
                      </span>
                      <div className="flex-1 h-3 rounded-full bg-black/[0.06] overflow-hidden">
                        <div className="bar-fill" style={{
                          width: `${(score as number).toFixed(1)}%`,
                          background: EMOTION_COLORS[label] ?? '#ccc',
                        }} />
                      </div>
                      <span className={`text-lg font-bold tabular-nums w-14 text-right ${isTop ? 'text-[#111]' : 'text-[#111] opacity-50'}`}>
                        {Math.round(score as number)}%
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="flex gap-1.5 flex-wrap mt-auto pt-4 border-t border-[#e4e4e4]">
                {Object.entries(result.modalities_used as Record<string, boolean>).map(([m, active]) => (
                  <span key={m} className={`text-xs font-semibold tracking-[0.05em] uppercase px-2.5 py-1 rounded-full border ${
                    active ? 'bg-[#111] text-white border-[#111]' : 'text-[#111] border-[#e4e4e4] opacity-30'
                  }`}>{m}</span>
                ))}
              </div>
            </div>

          </div>
        </div>
      )}
    </section>
  );
}
