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

export default function VideoSection() {
  const { addResult, isProcessing, setIsProcessing } = useEmotion();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoPreviewRef = useRef<HTMLVideoElement>(null);

  const [dragging, setDragging]     = useState(false);
  const [fileName, setFileName]     = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [file, setFile]             = useState<File | null>(null);
  const [textOverride, setTextOverride] = useState('');
  const [result, setResult]         = useState<any | null>(null);
  const [error, setError]           = useState<string | null>(null);

  const handleFile = (f: File) => {
    setFile(f);
    setFileName(f.name);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
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
      if (textOverride.trim()) {
        form.append('text', textOverride.trim());
      }

      const res = await fetch(`${API_BASE}/api/analyze/video`, {
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
        source: 'upload',
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
    <section className="analysis-section" id="video">
      <div className="analysis-section__header">
        <span className="analysis-section__tag">
          <span className="analysis-section__dot" />
          Video Analysis
        </span>
        <h2 className="section-title">
          Upload<br />
          <span className="italic">Video File</span>
        </h2>
        <p className="body-text" style={{ marginTop: 12 }}>
          Supports MP4, MOV, WebM, AVI, MKV. Full multimodal pipeline: ffmpeg extracts
          audio → opensmile COVAREP + py-feat OpenFace AU + Whisper transcript → fusion model.
        </p>
      </div>

      <div className="analysis-section__body">
        {/* Drop zone / preview */}
        <div
          className={`dropzone dropzone--video${dragging ? ' dropzone--active' : ''}${previewUrl ? ' dropzone--preview' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => !previewUrl && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*,.mp4,.mov,.webm,.avi,.mkv,.m4v,.3gp"
            style={{ display: 'none' }}
            onChange={onInputChange}
          />

          {previewUrl ? (
            <div className="dropzone__video-wrap">
              <video
                ref={videoPreviewRef}
                src={previewUrl}
                muted
                playsInline
                controls
                className="dropzone__video"
              />
              <button
                className="dropzone__replace"
                onClick={e => { e.stopPropagation(); fileInputRef.current?.click(); }}
              >
                Replace file
              </button>
            </div>
          ) : (
            <>
              <div className="dropzone__icon">
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                  <rect x="2" y="8" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="1.5" fill="none"/>
                  <path d="M22 13l8-5v16l-8-5V13z" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinejoin="round"/>
                </svg>
              </div>
              <p className="dropzone__label">Drop video file here</p>
              <p className="dropzone__hint">or click to browse · max 200 MB</p>
            </>
          )}
        </div>

        {/* Optional text override */}
        <div className="field-group" style={{ marginTop: 12 }}>
          <label className="field-label">
            Manual transcript <span className="field-label__optional">(optional — overrides Whisper)</span>
          </label>
          <input
            className="field-input"
            type="text"
            placeholder="Type the utterance if speech recognition is unavailable…"
            value={textOverride}
            onChange={e => setTextOverride(e.target.value)}
          />
        </div>

        {fileName && (
          <p className="dropzone__name" style={{ marginTop: 8 }}>{fileName}</p>
        )}

        <button
          className="btn btn-primary"
          onClick={analyze}
          disabled={!file || isProcessing}
          style={{ width: '100%', marginTop: 12 }}
        >
          {isProcessing ? 'Processing video…' : 'Analyse Video →'}
        </button>

        {isProcessing && (
          <p className="analysis-hint">
            Full pipeline may take 15–60 s depending on video length and server load.
          </p>
        )}

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