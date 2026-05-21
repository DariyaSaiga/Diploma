'use client';

import React, { useRef, useState, useCallback } from 'react';
import { useEmotion } from '@/lib/EmotionContex';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

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

type CsvSlot = 'audio' | 'vision';

function CsvDropSlot({ slot, label, hint, file, onFile }:
  { slot: CsvSlot; label: string; hint: string; file: File | null; onFile: (f: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0]; if (f) onFile(f);
  }, [onFile]);

  return (
    <div
      className={`flex items-center gap-3.5 border-[1.5px] rounded-xl px-4 py-4 cursor-pointer select-none transition-all duration-200 ${
        dragging ? 'border-red-400 bg-red-500/[0.06] border-solid'
        : file ? 'border-solid border-[#111] bg-black/[0.02]'
        : 'border-dashed border-[#d4d4d4] hover:border-red-400 hover:bg-red-500/[0.06]'
      }`}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept=".csv,text/csv" className="hidden"
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
      <div className={`shrink-0 text-[#111] ${file ? 'opacity-80' : 'opacity-40'}`}>
        {slot === 'audio' ? (
          <svg width="22" height="22" viewBox="0 0 20 20" fill="none">
            <rect x="2" y="12" width="2" height="5" rx="1" fill="currentColor" opacity=".4"/>
            <rect x="6" y="8" width="2" height="9" rx="1" fill="currentColor" opacity=".6"/>
            <rect x="10" y="4" width="2" height="13" rx="1" fill="currentColor"/>
            <rect x="14" y="7" width="2" height="10" rx="1" fill="currentColor" opacity=".7"/>
            <rect x="18" y="10" width="2" height="7" rx="1" fill="currentColor" opacity=".4"/>
          </svg>
        ) : (
          <svg width="22" height="22" viewBox="0 0 20 20" fill="none">
            <circle cx="10" cy="10" r="3.5" stroke="currentColor" strokeWidth="1.5"/>
            <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="1" opacity=".4"/>
            <path d="M10 3v2M10 15v2M3 10h2M15 10h2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        )}
      </div>
      <div className="flex flex-col gap-0.5 flex-1 min-w-0">
        <span className="text-base font-semibold text-[#111]">{label}</span>
        <span className="text-sm text-[#111] opacity-45 whitespace-nowrap overflow-hidden text-ellipsis">
          {file ? file.name : hint}
        </span>
      </div>
      {file && <span className="text-base text-[#2d8a4e] shrink-0">✓</span>}
    </div>
  );
}

export default function CsvSection() {
  const { addResult } = useEmotion();
  const [audioFile,       setAudioFile]       = useState<File | null>(null);
  const [visionFile,      setVisionFile]      = useState<File | null>(null);
  const [text,            setText]            = useState('');
  const [result,          setResult]          = useState<any | null>(null);
  const [error,           setError]           = useState<string | null>(null);
  const [localProcessing, setLocalProcessing] = useState(false);

  const analyze = async () => {
    if (!audioFile && !visionFile && !text.trim()) { setError('Provide at least one CSV file or a text utterance.'); return; }
    setLocalProcessing(true); setError(null); setResult(null);
    try {
      const form = new FormData();
      if (audioFile)   form.append('audio_csv',  audioFile);
      if (visionFile)  form.append('vision_csv', visionFile);
      if (text.trim()) form.append('text', text.trim());
      const res = await fetch(`${API_BASE}/api/analyze/multimodal/csv`, { method: 'POST', body: form });
      if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail ?? `HTTP ${res.status}`); }
      const data = await res.json();
      setResult(data);
      addResult({ emotion: data.emotion, confidence: data.confidence,
        allEmotions: Object.entries(data.probabilities as Record<string, number>).map(([label, score]) => ({ label, score })),
        source: 'upload', timestamp: new Date() });
    } catch (e: any) { setError(e.message ?? 'Unknown error'); }
    finally { setLocalProcessing(false); }
  };

  const topEmotion = result
    ? Object.entries(result.probabilities as Record<string, number>).sort((a, b) => b[1] - a[1])
    : [];
  const canSubmit = !localProcessing && (!!audioFile || !!visionFile || !!text.trim());

  return (
    <section className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-x-16 px-12 py-20 max-md:px-5 max-md:py-12 border-t border-[#e8e8e8] items-start" id="csv">

      {/* Col 1: header */}
      <div>
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold tracking-[0.08em] uppercase text-[#111] opacity-50 mb-4">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-current" />
          CSV / Research Mode
        </span>
        <h2 className="font-display font-black tracking-[-0.03em] leading-[1.1] mb-6" style={{ fontSize: 'clamp(28px, 4vw, 48px)' }}>
          Upload<br /><span className="italic font-normal">Feature CSVs</span>
        </h2>
        <p className="text-base lg:text-lg text-[#666] leading-[1.75] mt-3">
          For research pipelines with pre-extracted features. Upload COVAREP audio CSV
          (74 columns) and / or OpenFace AU vision CSV (35 columns). Non-numeric columns
          are stripped automatically.
        </p>
      </div>

      {/* Col 2: body + result */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-2.5">
          <CsvDropSlot slot="audio" label="Audio features CSV" hint="COVAREP · 74 dim · drop or click" file={audioFile} onFile={setAudioFile} />
          <CsvDropSlot slot="vision" label="Vision features CSV" hint="OpenFace AU · 35 dim · drop or click" file={visionFile} onFile={setVisionFile} />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold tracking-[0.04em] uppercase text-[#111] opacity-50">
            Utterance text{' '}
            <span className="font-normal normal-case tracking-normal opacity-70">(optional)</span>
          </label>
          <input type="text" placeholder="Type transcript for BERT text branch…" value={text} onChange={e => setText(e.target.value)}
            className="w-full px-3.5 py-3 text-base border-[1.5px] border-[#d4d4d4] rounded-lg bg-transparent text-[#111] outline-none focus:border-[#111] transition-colors duration-200 placeholder:opacity-35 box-border" />
        </div>

        <button
          className="inline-flex items-center justify-center w-full py-3.5 px-6 text-base font-semibold rounded-lg border-none bg-[#111] text-white cursor-pointer hover:opacity-80 transition-opacity duration-150 disabled:opacity-30 disabled:cursor-not-allowed"
          onClick={analyze} disabled={!canSubmit}
        >
          {localProcessing ? 'Running inference…' : 'Run Multimodal Inference →'}
        </button>

        {error && (
          <div className="flex items-start gap-2 p-3.5 bg-[#fff1f1] border border-[#fcc] rounded-lg text-base text-[#c00]">
            <span>⚠</span> {error}
          </div>
        )}
      </div>

      {result && (
        <div className="md:col-span-2 border-[1.5px] border-[#e4e4e4] rounded-2xl overflow-hidden mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2">
            {/* Left: dominant emotion */}
            <div className="flex flex-col items-center justify-center gap-4 p-10 text-center"
              style={{ background: EMOTION_COLORS[result.emotion] ?? '#f0f0f0' }}>
              <span className="font-goldman font-black capitalize leading-none text-[#111]"
                style={{ fontSize: 'clamp(40px, 6vw, 80px)' }}>{result.emotion}</span>
              <span className="font-goldman font-black leading-none tabular-nums text-[#111]"
                style={{ fontSize: 'clamp(32px, 4vw, 60px)' }}>{result.confidence.toFixed(2)}%</span>
              <span className="text-xs text-[#111] opacity-40">{result.latency_ms} ms</span>
            </div>

            {/* Right: all emotions */}
            <div className="p-8 flex flex-col gap-4 border-t md:border-t-0 md:border-l border-[#e4e4e4] bg-white">
              <p className="text-sm font-bold tracking-[0.1em] uppercase text-[#111] opacity-40">All emotions</p>
              {topEmotion.map(([label, score]) => {
                const isTop = label === result.emotion;
                return (
                  <div key={label} className="flex items-center gap-3">
                    <span className={`text-lg capitalize text-[#111] w-24 shrink-0 ${isTop ? 'font-bold' : 'font-medium opacity-70'}`}>{label}</span>
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
              {result.modalities_used && (
                <div className="flex gap-1.5 flex-wrap mt-2 pt-3 border-t border-[#e4e4e4]">
                  {Object.entries(result.modalities_used as Record<string, boolean>).map(([m, active]) => (
                    <span key={m} className={`text-xs font-semibold tracking-[0.05em] uppercase px-2.5 py-1 rounded-full border ${
                      active ? 'bg-[#111] text-white border-[#111]' : 'text-[#111] border-[#e4e4e4] opacity-30'
                    }`}>{m}</span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
