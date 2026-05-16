"use client";

import { useState, useRef, useEffect, useCallback } from "react";

// ─── Color Palette (Moon Spell – easy to swap) ───────────────────────────────
// Primary:   #7b5293  (Velvet Purple)
// Secondary: #d9c494  (Moon Sand / gold)
// Accent1:   #edabf2  (Pink Mist)
// Accent2:   #bbedef  (Icy Blue)
// Base:      #fbe3f6  (Blush Milk)
// ─────────────────────────────────────────────────────────────────────────────

const EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"];
const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/camera";
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type EmotionResult = {
  dominant: string;
  scores: Record<string, number>;
  timestamp?: number;
};

type ToastType = "success" | "error" | "info";
type Toast = { id: number; msg: string; type: ToastType };

// ─── Utility ─────────────────────────────────────────────────────────────────
function emotionEmoji(e: string) {
  const m: Record<string, string> = {
    Happy: "😄", Sad: "😢", Angry: "😠", Fear: "😨",
    Surprise: "😲", Disgust: "🤢", Neutral: "😐",
  };
  return m[e] ?? "🤖";
}

function emotionColor(e: string) {
  const m: Record<string, string> = {
    Happy: "#d9c494", Sad: "#bbedef", Angry: "#f87171",
    Fear: "#edabf2", Surprise: "#7b5293", Disgust: "#6ee7b7", Neutral: "#94a3b8",
  };
  return m[e] ?? "#edabf2";
}

// ─── Toast Component ──────────────────────────────────────────────────────────
function ToastList({ toasts, remove }: { toasts: Toast[]; remove: (id: number) => void }) {
  return (
    <div className="fixed top-6 right-6 z-50 flex flex-col gap-3 pointer-events-none">
      {toasts.map((t) => (
        <div
          key={t.id}
          onClick={() => remove(t.id)}
          className={`pointer-events-auto flex items-center gap-3 px-5 py-3 rounded-xl shadow-2xl backdrop-blur-xl border text-sm font-medium cursor-pointer
            transition-all duration-300 animate-slide-in
            ${t.type === "success" ? "bg-[#7b5293]/30 border-[#edabf2]/40 text-[#edabf2]"
              : t.type === "error" ? "bg-red-900/30 border-red-400/40 text-red-300"
              : "bg-[#1a1035]/60 border-[#bbedef]/30 text-[#bbedef]"}`}
        >
          <span>{t.type === "success" ? "✓" : t.type === "error" ? "✕" : "ℹ"}</span>
          {t.msg}
        </div>
      ))}
    </div>
  );
}

// ─── Skeleton Loader ──────────────────────────────────────────────────────────
function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse bg-white/5 rounded-xl ${className}`} />;
}

// ─── Emotion Bar ──────────────────────────────────────────────────────────────
function EmotionBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-20 text-xs text-white/50 text-right shrink-0">{label}</span>
      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${(value * 100).toFixed(1)}%`, background: color, boxShadow: `0 0 8px ${color}80` }}
        />
      </div>
      <span className="w-10 text-xs text-white/40 shrink-0">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

// ─── Results Panel ────────────────────────────────────────────────────────────
function ResultsPanel({ result, loading }: { result: EmotionResult | null; loading: boolean }) {
  if (loading) {
    return (
      <div className="glass-card rounded-xl p-6 space-y-4">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-16 w-full" />
        {[...Array(7)].map((_, i) => <Skeleton key={i} className="h-3 w-full" />)}
      </div>
    );
  }

  if (!result) {
    return (
      <div className="glass-card rounded-xl p-8 flex flex-col items-center justify-center gap-4 min-h-[220px] border border-white/5">
        <div className="w-14 h-14 rounded-full bg-[#7b5293]/20 flex items-center justify-center text-2xl">🤖</div>
        <p className="text-white/30 text-sm text-center">Results will appear here after detection</p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-xl p-6 space-y-5">
      {/* Dominant emotion hero */}
      <div className="flex items-center gap-4 p-4 rounded-xl" style={{ background: `${emotionColor(result.dominant)}15`, border: `1px solid ${emotionColor(result.dominant)}40` }}>
        <span className="text-4xl">{emotionEmoji(result.dominant)}</span>
        <div>
          <p className="text-xs text-white/40 uppercase tracking-widest">Dominant Emotion</p>
          <p className="text-2xl font-bold" style={{ color: emotionColor(result.dominant) }}>{result.dominant}</p>
          <p className="text-xs text-white/30">
            Confidence: {((result.scores[result.dominant] ?? 0) * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* All emotion bars */}
      <div className="space-y-2.5">
        <p className="text-xs text-white/30 uppercase tracking-widest mb-3">Emotion Probabilities</p>
        {EMOTIONS.map((e) => (
          <EmotionBar key={e} label={e} value={result.scores[e] ?? 0} color={emotionColor(e)} />
        ))}
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function Home() {
  // Toast state
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastIdRef = useRef(0);
  const addToast = useCallback((msg: string, type: ToastType = "info") => {
    const id = ++toastIdRef.current;
    setToasts((p) => [...p, { id, msg, type }]);
    setTimeout(() => setToasts((p) => p.filter((t) => t.id !== id)), 4000);
  }, []);

  // ── Camera detection state
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [camActive, setCamActive] = useState(false);
  const [camResult, setCamResult] = useState<EmotionResult | null>(null);
  const [camLoading, setCamLoading] = useState(false);

  // ── Upload state
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<EmotionResult | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ─── Camera: start / stop ─────────────────────────────────────────────────
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      // Connect WebSocket
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => addToast("Camera detection started", "success");
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data) as EmotionResult;
        setCamResult(data);
        setCamLoading(false);
      };
      ws.onerror = () => addToast("WebSocket error", "error");
      ws.onclose = () => setCamActive(false);

      // Send frames every 500ms
      intervalRef.current = setInterval(() => {
        if (!videoRef.current || !canvasRef.current) return;
        const ctx = canvasRef.current.getContext("2d");
        if (!ctx) return;
        canvasRef.current.width = 224;
        canvasRef.current.height = 224;
        ctx.drawImage(videoRef.current, 0, 0, 224, 224);
        const frame = canvasRef.current.toDataURL("image/jpeg", 0.7).split(",")[1];
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ frame }));
      }, 500);

      setCamActive(true);
      setCamLoading(true);
    } catch {
      addToast("Camera access denied", "error");
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    wsRef.current?.close();
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setCamActive(false);
    addToast("Camera stopped", "info");
  };

  useEffect(() => () => stopCamera(), []); // cleanup on unmount

  // ─── Upload: drag & drop + file select ───────────────────────────────────
  const handleFile = (file: File) => {
    if (!file.type.startsWith("video/")) { addToast("Please upload a video file", "error"); return; }
    setVideoFile(file);
    setVideoPreview(URL.createObjectURL(file));
    setUploadResult(null);
    addToast(`${file.name} ready for analysis`, "success");
  };

  const analyzeVideo = async () => {
    if (!videoFile) return;
    setUploadLoading(true);
    const form = new FormData();
    form.append("file", videoFile);
    try {
      const res = await fetch(`${API_URL}/analyze/video`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json() as EmotionResult;
      setUploadResult(data);
      addToast("Analysis complete!", "success");
    } catch (e: unknown) {
      addToast(e instanceof Error ? e.message : "Analysis failed", "error");
    } finally {
      setUploadLoading(false);
    }
  };

  // ─── Navbar scroll effect ─────────────────────────────────────────────────
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", h);
    return () => window.removeEventListener("scroll", h);
  }, []);

  return (
    <>
      {/* Global CSS injected via globals.css – see that file */}
      <ToastList toasts={toasts} remove={(id) => setToasts((p) => p.filter((t) => t.id !== id))} />

      {/* ── NAVBAR ─────────────────────────────────────────────────────────── */}
      <nav className={`fixed top-0 inset-x-0 z-40 transition-all duration-300 ${scrolled ? "backdrop-blur-xl bg-[#0d0a1e]/80 border-b border-white/5" : "bg-transparent"}`}>
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#7b5293] to-[#edabf2] flex items-center justify-center text-sm">🧠</div>
            <span className="font-bold text-white tracking-tight">EmotionAI</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm text-white/40">
            <a href="#detection" className="hover:text-white transition-colors">Detection</a>
            <a href="#upload" className="hover:text-white transition-colors">Upload</a>
            <a href="#results" className="hover:text-white transition-colors">Results</a>
          </div>
          <a href="https://github.com" target="_blank" rel="noreferrer"
            className="btn-glow text-xs px-4 py-2 rounded-xl font-medium text-white">
            GitHub ↗
          </a>
        </div>
      </nav>

      <main className="min-h-screen bg-[#0d0a1e] text-white overflow-x-hidden">

        {/* ── BACKGROUND ORBS ───────────────────────────────────────────────── */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
          <div className="orb w-[600px] h-[600px] bg-[#7b5293] top-[-200px] left-[-200px] opacity-20" />
          <div className="orb w-[500px] h-[500px] bg-[#edabf2] bottom-[-150px] right-[-150px] opacity-10" />
          <div className="orb w-[300px] h-[300px] bg-[#bbedef] top-[40%] left-[50%] opacity-8" />
        </div>

        {/* ── HERO SECTION ──────────────────────────────────────────────────── */}
        <section className="relative pt-32 pb-20 px-6 text-center max-w-5xl mx-auto">

          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-[#7b5293]/40 bg-[#7b5293]/10 text-[#edabf2] text-xs mb-8 animate-fade-up">
            <span className="w-1.5 h-1.5 rounded-full bg-[#edabf2] animate-pulse" />
            Multimodal AI · Attention Bottleneck Mechanism
          </div>

          <h1 className="text-5xl md:text-6xl font-black leading-tight tracking-tight mb-6 animate-fade-up" style={{ animationDelay: "0.1s" }}>
            <span className="text-white">Emotion</span>
            <span className="bg-gradient-to-r from-[#7b5293] via-[#edabf2] to-[#bbedef] bg-clip-text text-transparent"> Recognition</span>
            <br />
            <span className="text-white/60 text-4xl md:text-5xl font-light">System</span>
          </h1>

          <p className="text-lg text-white/40 max-w-2xl mx-auto mb-10 animate-fade-up" style={{ animationDelay: "0.2s" }}>
            A deep learning system fusing visual and audio signals with a cross-modal attention bottleneck to infer human emotional states in real time.
          </p>

          {/* Stats row */}
          <div className="flex flex-wrap justify-center gap-4 animate-fade-up" style={{ animationDelay: "0.3s" }}>
            {[
              { label: "Emotions Detected", value: "7" },
              { label: "Modalities", value: "2" },
              { label: "Inference Speed", value: "~30ms" },
              { label: "Architecture", value: "ABM" },
            ].map((s) => (
              <div key={s.label} className="stat-card glass-card rounded-xl px-6 py-3 text-center">
                <p className="text-xl font-bold bg-gradient-to-r from-[#d9c494] to-[#edabf2] bg-clip-text text-transparent">{s.value}</p>
                <p className="text-xs text-white/30">{s.label}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── MAIN CONTENT GRID ─────────────────────────────────────────────── */}
        <div className="max-w-7xl mx-auto px-6 pb-24 space-y-10">

          {/* ── ROW 1: Camera + Upload ──────────────────────────────────────── */}
          <div className="grid md:grid-cols-2 gap-6" id="detection">

            {/* ── CAMERA CARD ───────────────────────────────────────────────── */}
            <div className="glass-card rounded-xl p-6 space-y-4 card-hover">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${camActive ? "bg-green-400 animate-pulse" : "bg-white/20"}`} />
                    <h2 className="font-bold text-lg text-white">Live Camera</h2>
                  </div>
                  <p className="text-sm text-white/30 mt-0.5">Real-time emotion detection via webcam</p>
                </div>
                <div className="w-10 h-10 rounded-xl bg-[#7b5293]/20 flex items-center justify-center text-lg">📷</div>
              </div>

              {/* Video preview */}
              <div className="relative bg-black/40 rounded-xl overflow-hidden aspect-video flex items-center justify-center border border-white/5">
                <video ref={videoRef} className="w-full h-full object-cover" muted playsInline />
                <canvas ref={canvasRef} className="hidden" />
                {!camActive && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                    <div className="w-16 h-16 rounded-full bg-[#7b5293]/20 flex items-center justify-center text-3xl">📹</div>
                    <p className="text-white/20 text-sm">Camera preview</p>
                  </div>
                )}
                {/* Overlay HUD */}
                {camActive && camResult && (
                  <div className="absolute bottom-3 left-3 right-3 glass-card rounded-xl px-4 py-2 flex items-center justify-between">
                    <span className="text-sm font-bold" style={{ color: emotionColor(camResult.dominant) }}>
                      {emotionEmoji(camResult.dominant)} {camResult.dominant}
                    </span>
                    <span className="text-xs text-white/30">
                      {((camResult.scores[camResult.dominant] ?? 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
              </div>

              <button
                onClick={camActive ? stopCamera : startCamera}
                className={`w-full py-3 rounded-xl font-semibold text-sm transition-all duration-200 ${
                  camActive
                    ? "bg-red-500/20 border border-red-500/40 text-red-400 hover:bg-red-500/30"
                    : "btn-glow text-white"
                }`}
              >
                {camActive ? "⏹ Stop Detection" : "▶ Start Detection"}
              </button>
            </div>

            {/* ── UPLOAD CARD ───────────────────────────────────────────────── */}
            <div className="glass-card rounded-xl p-6 space-y-4 card-hover" id="upload">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="font-bold text-lg text-white">Upload Video</h2>
                  <p className="text-sm text-white/30 mt-0.5">Analyze a pre-recorded video file</p>
                </div>
                <div className="w-10 h-10 rounded-xl bg-[#d9c494]/10 flex items-center justify-center text-lg">🎬</div>
              </div>

              {/* Drop zone */}
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
                onClick={() => fileInputRef.current?.click()}
                className={`relative rounded-xl border-2 border-dashed transition-all duration-200 cursor-pointer flex flex-col items-center justify-center gap-3 aspect-video
                  ${dragOver ? "border-[#7b5293] bg-[#7b5293]/10" : "border-white/10 bg-black/20 hover:border-[#7b5293]/50 hover:bg-[#7b5293]/5"}`}
              >
                <input ref={fileInputRef} type="file" accept="video/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />

                {videoPreview ? (
                  <video src={videoPreview} className="w-full h-full object-cover rounded-xl" controls />
                ) : (
                  <>
                    <div className="w-14 h-14 rounded-xl bg-[#d9c494]/10 flex items-center justify-center text-3xl">⬆</div>
                    <div className="text-center">
                      <p className="text-white/50 text-sm font-medium">Drop video here or click to browse</p>
                      <p className="text-white/20 text-xs mt-1">MP4, MOV, AVI, WEBM supported</p>
                    </div>
                  </>
                )}
              </div>

              {videoFile && (
                <div className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/5 border border-white/5">
                  <span className="text-lg">🎞</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-white/70 truncate">{videoFile.name}</p>
                    <p className="text-xs text-white/25">{(videoFile.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                  <button onClick={() => { setVideoFile(null); setVideoPreview(null); setUploadResult(null); }} className="text-white/20 hover:text-red-400 transition-colors text-lg">✕</button>
                </div>
              )}

              <button
                onClick={analyzeVideo}
                disabled={!videoFile || uploadLoading}
                className={`w-full py-3 rounded-xl font-semibold text-sm transition-all duration-200
                  ${videoFile && !uploadLoading ? "btn-glow text-white" : "bg-white/5 text-white/20 cursor-not-allowed"}`}
              >
                {uploadLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                    Analyzing…
                  </span>
                ) : "🔍 Analyze Video"}
              </button>
            </div>
          </div>

          {/* ── ROW 2: Results ─────────────────────────────────────────────── */}
          <div id="results">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-1 h-6 rounded-full bg-gradient-to-b from-[#7b5293] to-[#edabf2]" />
              <h2 className="text-xl font-bold text-white">Analysis Results</h2>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {/* Camera results */}
              <div>
                <p className="text-xs text-white/30 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <span className={`w-1.5 h-1.5 rounded-full ${camActive ? "bg-green-400 animate-pulse" : "bg-white/10"}`} />
                  Live Camera
                </p>
                <ResultsPanel result={camResult} loading={camLoading && camActive} />
              </div>

              {/* Upload results */}
              <div>
                <p className="text-xs text-white/30 uppercase tracking-widest mb-3">Video Upload</p>
                <ResultsPanel result={uploadResult} loading={uploadLoading} />
              </div>
            </div>
          </div>

          {/* ── ROW 3: Architecture info ────────────────────────────────────── */}
          <div className="glass-card rounded-xl p-8">
            <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
              <span className="text-2xl">🧬</span> Attention Bottleneck Architecture
            </h2>
            <div className="grid sm:grid-cols-3 gap-4">
              {[
                { icon: "👁", title: "Visual Encoder", desc: "ResNet-50 backbone extracts spatial facial features from video frames at 30fps with temporal pooling." },
                { icon: "🔊", title: "Audio Encoder", desc: "Wav2Vec 2.0 processes raw audio into speech embeddings capturing prosody and vocal affect patterns." },
                { icon: "⚡", title: "Attention Bottleneck", desc: "Cross-modal attention layers fuse both modalities via shared bottleneck tokens for joint emotion inference." },
              ].map((c) => (
                <div key={c.title} className="p-5 rounded-xl bg-white/3 border border-white/5 space-y-2 hover:border-[#7b5293]/40 transition-colors duration-200">
                  <div className="w-10 h-10 rounded-xl bg-[#7b5293]/20 flex items-center justify-center text-xl">{c.icon}</div>
                  <h3 className="font-semibold text-white text-sm">{c.title}</h3>
                  <p className="text-xs text-white/30 leading-relaxed">{c.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── FOOTER ────────────────────────────────────────────────────────── */}
        <footer className="border-t border-white/5 py-10 px-6">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-white/20">
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#7b5293] to-[#edabf2] flex items-center justify-center text-xs">🧠</div>
              <span>EmotionAI — Diploma Project 2024</span>
            </div>
            <p>Multimodal Emotion Recognition · Attention Bottleneck Mechanism</p>
            <p>Built with Next.js + FastAPI</p>
          </div>
        </footer>
      </main>
    </>
  );
}
