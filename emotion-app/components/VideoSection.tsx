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

export default function VideoSection() {
  const { addResult } = useEmotion();
  const fileInputRef    = useRef<HTMLInputElement>(null);
  const videoPreviewRef = useRef<HTMLVideoElement>(null);
  const abortRef        = useRef<AbortController | null>(null);

  const [dragging,        setDragging]        = useState(false);
  const [fileName,        setFileName]        = useState<string | null>(null);
  const [previewUrl,      setPreviewUrl]      = useState<string | null>(null);
  const [file,            setFile]            = useState<File | null>(null);
  const [textOverride,    setTextOverride]    = useState('');
  const [result,          setResult]          = useState<any | null>(null);
  const [error,           setError]           = useState<string | null>(null);
  const [localProcessing, setLocalProcessing] = useState(false);

  const handleFile = (f: File) => {
    setFile(f); setFileName(f.name); setResult(null); setError(null);
    setPreviewUrl(URL.createObjectURL(f));
  };
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
      if (textOverride.trim()) form.append('text', textOverride.trim());
      const res = await fetch(`${API_BASE}/api/analyze/video`, { method: 'POST', body: form, signal: controller.signal });
      if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail ?? `HTTP ${res.status}`); }
      const data = await res.json();
      setResult(data);
      addResult({ emotion: data.emotion, confidence: data.confidence,
        allEmotions: Object.entries(data.probabilities as Record<string, number>).map(([label, score]) => ({ label, score })),
        source: 'upload', timestamp: new Date() });
    } catch (e: any) {
      if (e.name === 'AbortError') return;
      setError(e.message ?? 'Unknown error');
    } finally { setLocalProcessing(false); abortRef.current = null; }
  };

  const topEmotion = result
    ? Object.entries(result.probabilities as Record<string, number>).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <section className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-x-16 px-12 py-20 max-md:px-5 max-md:py-12 border-t border-[#e8e8e8] items-start" id="video">

      {/* Col 1: header */}
      <div>
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold tracking-[0.08em] uppercase text-[#111] opacity-50 mb-4">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-current" />
          Video Analysis
        </span>
        <h2 className="font-display font-black tracking-[-0.03em] leading-[1.1] mb-6" style={{ fontSize: 'clamp(28px, 4vw, 48px)' }}>
          Upload<br /><span className="italic font-normal">Video File</span>
        </h2>
        <p className="text-base lg:text-lg text-[#666] leading-[1.75] mt-3">
          Supports MP4, MOV, WebM, AVI, MKV. Full multimodal pipeline: ffmpeg extracts
          audio → opensmile COVAREP + py-feat OpenFace AU + Whisper transcript → fusion model.
        </p>
      </div>

      {/* Col 2: body + result */}
      <div className="flex flex-col gap-3">
        {/* Dropzone / preview */}
        <div
          className={`border-[1.5px] rounded-xl overflow-hidden cursor-pointer transition-all duration-200 ${
            previewUrl ? 'border-solid border-[#111] p-0 min-h-[200px]'
            : dragging ? 'border-red-400 bg-red-500/[0.06] flex flex-col items-center justify-center gap-2 px-6 py-9'
            : 'border-dashed border-[#d4d4d4] hover:border-red-400 hover:bg-red-500/[0.06] flex flex-col items-center justify-center gap-2 px-6 py-9 min-h-[200px] text-center select-none'
          }`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => !previewUrl && fileInputRef.current?.click()}
        >
          <input ref={fileInputRef} type="file" accept="video/*,.mp4,.mov,.webm,.avi,.mkv,.m4v,.3gp" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
          {previewUrl ? (
            <div className="relative w-full">
              <video ref={videoPreviewRef} src={previewUrl} muted playsInline controls
                className="block w-full max-h-[260px] object-contain rounded-xl" />
              <button
                className="absolute bottom-2 right-2 text-xs px-2.5 py-1 rounded-md border border-white/60 bg-black/55 text-white cursor-pointer backdrop-blur-sm"
                onClick={e => { e.stopPropagation(); fileInputRef.current?.click(); }}
              >
                Replace file
              </button>
            </div>
          ) : (
            <>
              <div className="text-[#111] opacity-35 mb-1">
                <svg width="36" height="36" viewBox="0 0 32 32" fill="none">
                  <rect x="2" y="8" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="1.5" fill="none"/>
                  <path d="M22 13l8-5v16l-8-5V13z" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinejoin="round"/>
                </svg>
              </div>
              <p className="text-base font-medium text-[#111]">Drop video file here</p>
              <p className="text-sm text-[#111] opacity-40">or click to browse · max 200 MB</p>
            </>
          )}
        </div>

        {/* Optional transcript */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold tracking-[0.04em] uppercase text-[#111] opacity-50">
            Manual transcript{' '}
            <span className="font-normal normal-case tracking-normal opacity-70">(optional — overrides Whisper)</span>
          </label>
          <input type="text" placeholder="Type the utterance if speech recognition is unavailable…"
            value={textOverride} onChange={e => setTextOverride(e.target.value)}
            className="w-full px-3.5 py-3 text-base border-[1.5px] border-[#d4d4d4] rounded-lg bg-transparent text-[#111] outline-none focus:border-[#111] transition-colors duration-200 placeholder:opacity-35 box-border" />
        </div>

        {fileName && <p className="text-base font-medium text-[#111] opacity-70 break-all">{fileName}</p>}

        <button
          className="inline-flex items-center justify-center w-full py-3.5 px-6 text-base font-semibold rounded-lg border-none bg-[#111] text-white cursor-pointer hover:opacity-80 transition-opacity duration-150 disabled:opacity-30 disabled:cursor-not-allowed"
          onClick={analyze} disabled={!file || localProcessing}
        >
          {localProcessing ? 'Processing video…' : 'Analyse Video →'}
        </button>

        {localProcessing && (
          <p className="text-sm text-[#111] opacity-45">
            Full pipeline may take 15–60 s depending on video length and server load.
          </p>
        )}

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
              {result.transcript && (
                <p className="text-sm opacity-60 italic text-[#111] max-w-xs">"{result.transcript}"</p>
              )}
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
