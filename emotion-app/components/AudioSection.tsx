'use client';

import React, { useRef, useState, useCallback } from 'react';
import { useEmotion } from '@/lib/EmotionContex';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

const EMOTION_COLORS: Record<string, string> = {
  happiness: '#d4f5c4',
  sadness:   '#c4d8f5',
  anger:     '#f5c4c4',
  surprise:  '#f5ecc4',
  disgust:   '#d8c4f5',
  fear:      '#f5d4c4',
};

export default function AudioSection() {
  const { addResult, isProcessing, setIsProcessing } = useEmotion();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging]   = useState(false);
  const [fileName, setFileName]   = useState<string | null>(null);
  const [file, setFile]           = useState<File | null>(null);
  const [result, setResult]       = useState<any | null>(null);
  const [error, setError]         = useState<string | null>(null);

  const handleFile = (f: File) => {
    setFile(f);
    setFileName(f.name);
    setResult(null);
    setError(null);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append('file', file);

      const res = await fetch(`${API_BASE}/api/analyze/audio`, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);

      addResult({
        emotion: data.emotion,
        confidence: data.confidence,
        allEmotions: Object.entries(data.probabilities as Record<string, number>).map(
          ([label, score]) => ({ label, score })
        ),
        source: 'audio',
        timestamp: new Date(),
      });
    } catch (e: any) {
      setError(e.message ?? 'Unknown error');
    } finally {
      setIsProcessing(false);
    }
  };

  const topEmotion = result
    ? Object.entries(result.probabilities as Record<string, number>)
        .sort((a, b) => b[1] - a[1])
    : [];

  return (
    <section className="analysis-section" id="audio">
      <div className="analysis-section__header">
        <span className="analysis-section__tag">
          <span className="analysis-section__dot" />
          Audio Analysis
        </span>
        <h2 className="section-title">
          Upload<br />
          <span className="italic">Audio File</span>
        </h2>
        <p className="body-text" style={{ marginTop: 12 }}>
          Supports WAV, MP3, M4A, FLAC, OGG. The model extracts COVAREP acoustic
          features via opensmile and transcribes speech with Whisper for joint
          audio + text inference.
        </p>
      </div>

      <div className="analysis-section__body">
        {/* Drop zone */}
        <div
          className={`dropzone${dragging ? ' dropzone--active' : ''}${fileName ? ' dropzone--filled' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.aac,.opus,.wma"
            style={{ display: 'none' }}
            onChange={onInputChange}
          />

          <div className="dropzone__icon">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <rect x="4" y="20" width="3" height="8" rx="1.5" fill="currentColor" opacity=".4"/>
              <rect x="9" y="14" width="3" height="14" rx="1.5" fill="currentColor" opacity=".6"/>
              <rect x="14" y="8"  width="3" height="20" rx="1.5" fill="currentColor"/>
              <rect x="19" y="12" width="3" height="16" rx="1.5" fill="currentColor" opacity=".7"/>
              <rect x="24" y="17" width="3" height="11" rx="1.5" fill="currentColor" opacity=".4"/>
            </svg>
          </div>

          {fileName ? (
            <p className="dropzone__name">{fileName}</p>
          ) : (
            <>
              <p className="dropzone__label">Drop audio file here</p>
              <p className="dropzone__hint">or click to browse</p>
            </>
          )}
        </div>

        <button
          className="btn btn-primary"
          onClick={analyze}
          disabled={!file || isProcessing}
          style={{ width: '100%', marginTop: 12 }}
        >
          {isProcessing ? 'Analysing…' : 'Analyse Audio →'}
        </button>

        {error && (
          <div className="analysis-error">
            <span>⚠</span> {error}
          </div>
        )}
      </div>

      {/* Result panel */}
      {result && (
        <div className="result-panel">
          <div className="result-panel__header">
            <span
              className="result-panel__emotion-badge"
              style={{ background: EMOTION_COLORS[result.emotion] ?? '#eee' }}
            >
              {result.emotion}
            </span>
            <span className="result-panel__confidence">
              {result.confidence.toFixed(1)}%
            </span>
            <span className="result-panel__latency">{result.latency_ms} ms</span>
          </div>

          {result.transcript && (
            <div className="result-panel__transcript">
              <span className="result-panel__transcript-label">Transcript</span>
              <p className="result-panel__transcript-text">"{result.transcript}"</p>
            </div>
          )}

          <div className="result-panel__bars">
            {topEmotion.map(([label, score]) => (
              <div className="result-bar" key={label}>
                <span className="result-bar__label">{label}</span>
                <div className="result-bar__track">
                  <div
                    className="result-bar__fill"
                    style={{
                      width: `${(score * 100).toFixed(1)}%`,
                      background: EMOTION_COLORS[label] ?? '#ccc',
                    }}
                  />
                </div>
                <span className="result-bar__value">{(score * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>

          <div className="result-panel__modalities">
            {Object.entries(result.modalities_used as Record<string, boolean>).map(([m, active]) => (
              <span key={m} className={`modality-tag${active ? ' modality-tag--active' : ''}`}>
                {m}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}