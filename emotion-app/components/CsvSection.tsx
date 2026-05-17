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

type CsvSlot = 'audio' | 'vision';

function CsvDropSlot({
  slot,
  label,
  hint,
  file,
  onFile,
}: {
  slot: CsvSlot;
  label: string;
  hint: string;
  file: File | null;
  onFile: (f: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) onFile(f);
    },
    [onFile]
  );

  return (
    <div
      className={`csv-slot${dragging ? ' csv-slot--active' : ''}${file ? ' csv-slot--filled' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        style={{ display: 'none' }}
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); }}
      />

      <div className="csv-slot__icon">
        {slot === 'audio' ? (
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <rect x="2" y="12" width="2" height="5" rx="1" fill="currentColor" opacity=".4"/>
            <rect x="6" y="8"  width="2" height="9" rx="1" fill="currentColor" opacity=".6"/>
            <rect x="10" y="4" width="2" height="13" rx="1" fill="currentColor"/>
            <rect x="14" y="7" width="2" height="10" rx="1" fill="currentColor" opacity=".7"/>
            <rect x="18" y="10" width="2" height="7"  rx="1" fill="currentColor" opacity=".4"/>
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <circle cx="10" cy="10" r="3.5" stroke="currentColor" strokeWidth="1.5"/>
            <circle cx="10" cy="10" r="7"   stroke="currentColor" strokeWidth="1"  opacity=".4"/>
            <path d="M10 3v2M10 15v2M3 10h2M15 10h2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        )}
      </div>

      <div className="csv-slot__text">
        <span className="csv-slot__label">{label}</span>
        <span className="csv-slot__hint">
          {file ? file.name : hint}
        </span>
      </div>

      {file && <span className="csv-slot__check">✓</span>}
    </div>
  );
}

export default function CsvSection() {
  const { addResult, isProcessing, setIsProcessing } = useEmotion();

  const [audioFile, setAudioFile]   = useState<File | null>(null);
  const [visionFile, setVisionFile] = useState<File | null>(null);
  const [text, setText]             = useState('');
  const [result, setResult]         = useState<any | null>(null);
  const [error, setError]           = useState<string | null>(null);

  const analyze = async () => {
    if (!audioFile && !visionFile && !text.trim()) {
      setError('Provide at least one CSV file or a text utterance.');
      return;
    }
    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      if (audioFile)    form.append('audio_csv',  audioFile);
      if (visionFile)   form.append('vision_csv', visionFile);
      if (text.trim())  form.append('text', text.trim());

      const res = await fetch(`${API_BASE}/api/analyze/multimodal/csv`, {
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

  const canSubmit = !isProcessing && (!!audioFile || !!visionFile || !!text.trim());

  return (
    <section className="analysis-section" id="csv">
      <div className="analysis-section__header">
        <span className="analysis-section__tag">
          <span className="analysis-section__dot" />
          CSV / Research Mode
        </span>
        <h2 className="section-title">
          Upload<br />
          <span className="italic">Feature CSVs</span>
        </h2>
        <p className="body-text" style={{ marginTop: 12 }}>
          For research pipelines with pre-extracted features. Upload COVAREP audio CSV
          (74 columns) and / or OpenFace AU vision CSV (35 columns). Non-numeric columns
          are stripped automatically.
        </p>
      </div>

      <div className="analysis-section__body">
        <div className="csv-slots">
          <CsvDropSlot
            slot="audio"
            label="Audio features CSV"
            hint="COVAREP · 74 dim · drop or click"
            file={audioFile}
            onFile={setAudioFile}
          />
          <CsvDropSlot
            slot="vision"
            label="Vision features CSV"
            hint="OpenFace AU · 35 dim · drop or click"
            file={visionFile}
            onFile={setVisionFile}
          />
        </div>

        <div className="field-group" style={{ marginTop: 12 }}>
          <label className="field-label">
            Utterance text <span className="field-label__optional">(optional)</span>
          </label>
          <input
            className="field-input"
            type="text"
            placeholder="Type transcript for BERT text branch…"
            value={text}
            onChange={e => setText(e.target.value)}
          />
        </div>

        <button
          className="btn btn-primary"
          onClick={analyze}
          disabled={!canSubmit}
          style={{ width: '100%', marginTop: 12 }}
        >
          {isProcessing ? 'Running inference…' : 'Run Multimodal Inference →'}
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