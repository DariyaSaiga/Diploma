import { useState, useEffect, useRef } from "react";

const EMOTIONS = [
  { label: "Joy",      color: "#F4C430", x: 50, y: 30 },
  { label: "Sadness",  color: "#4A90D9", x: 20, y: 65 },
  { label: "Anger",    color: "#E84040", x: 80, y: 65 },
  { label: "Fear",     color: "#9B59B6", x: 35, y: 85 },
  { label: "Surprise", color: "#1ABC9C", x: 65, y: 85 },
  { label: "Disgust",  color: "#E67E22", x: 50, y: 70 },
];

const MODALITIES = [
  { name: "Audio",  icon: "🎙", desc: "Prosody, pitch, MFCCs",           color: "#4A90D9" },
  { name: "Visual", icon: "👁", desc: "Facial Action Units, landmarks",   color: "#E84040" },
  { name: "Text",   icon: "✦", desc: "Semantic embeddings, syntax",      color: "#1ABC9C" },
];

const TIMELINE = [
  { phase: "Data collection",      desc: "CMU-MOSEI dataset",        icon: "∷" },
  { phase: "Feature extraction",   desc: "Audio 1DCNN · Visual OpenFace · Text BERT", icon: "◈" },
  { phase: "Attention bottleneck", desc: "Feature fusion (text + audio + video)",     icon: "⬡" },
  { phase: "Classification",       desc: "6-class emotion output with confidence",   icon: "◎" },
];

function fakeAnalyze() {
  const base = [
    { label: "Joy",      color: "#F4C430" },
    { label: "Sadness",  color: "#4A90D9" },
    { label: "Anger",    color: "#E84040" },
    { label: "Fear",     color: "#9B59B6" },
    { label: "Surprise", color: "#1ABC9C" },
    { label: "Disgust",  color: "#E67E22" },
  ];
  const raw = base.map(e => ({ ...e, score: Math.random() * 100 }));
  const total = raw.reduce((s, e) => s + e.score, 0);
  return raw.map(e => ({ ...e, score: Math.round((e.score / total) * 100) })).sort((a, b) => b.score - a.score);
}

/* ── Particle canvas ── */
function ParticleField() {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d");
    const resize = () => { c.width = c.offsetWidth; c.height = c.offsetHeight; };
    resize(); window.addEventListener("resize", resize);
    const pts = Array.from({ length: 55 }, () => ({
      x: Math.random() * c.width, y: Math.random() * c.height,
      vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35,
      r: Math.random() * 1.8 + 0.4, a: Math.random() * 0.45 + 0.1,
    }));
    let raf;
    const draw = () => {
      ctx.clearRect(0, 0, c.width, c.height);
      pts.forEach((p, i) => {
        p.x = (p.x + p.vx + c.width) % c.width;
        p.y = (p.y + p.vy + c.height) % c.height;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(100,160,255,${p.a})`; ctx.fill();
        pts.slice(i + 1).forEach(q => {
          const d = Math.hypot(p.x - q.x, p.y - q.y);
          if (d < 110) { ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.strokeStyle = `rgba(100,160,255,${0.07*(1-d/110)})`; ctx.lineWidth = 0.5; ctx.stroke(); }
        });
      });
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize); };
  }, []);
  return <canvas ref={ref} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} />;
}

/* ── Animated counter ── */
function Counter({ target, suffix = "" }) {
  const [v, setV] = useState(0); const ref = useRef(null);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      let cur = 0; const step = target / 60;
      const t = setInterval(() => { cur += step; if (cur >= target) { setV(target); clearInterval(t); } else setV(Math.floor(cur)); }, 16);
    }, { threshold: 0.5 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, [target]);
  return <span ref={ref}>{v}{suffix}</span>;
}

/* ── Attention viz ── */
function AttentionViz() {
  const [active, setActive] = useState(null);
  const DESCS = ["cross-modal","temporal","semantic","prosodic","visual-text","audio-visual","global","local"];
  return (
    <div style={{ fontFamily: "'Space Mono',monospace", padding: "2rem 0" }}>
      <div style={{ textAlign:"center", marginBottom:"1.5rem", fontSize:13, color:"#8899aa", letterSpacing:2, textTransform:"uppercase" }}>
        Attention Bottleneck — 8 heads
      </div>
      <div style={{ display:"flex", justifyContent:"center", gap:"2rem", marginBottom:"1.5rem", flexWrap:"wrap" }}>
        {["Audio","Visual","Text"].map((s, si) => (
          <div key={s} style={{ padding:"8px 20px", border:`1px solid ${["#4A90D9","#E84040","#1ABC9C"][si]}44`, borderRadius:4, fontSize:12, color:["#4A90D9","#E84040","#1ABC9C"][si], letterSpacing:1 }}>{s}</div>
        ))}
      </div>
      <div style={{ display:"flex", justifyContent:"center", gap:8, marginBottom:"1.5rem", flexWrap:"wrap" }}>
        {[0,1,2,3,4,5,6,7].map(h => (
          <div key={h} onMouseEnter={() => setActive(h)} onMouseLeave={() => setActive(null)}
            style={{ width:40, height:40, borderRadius:4, background:active===h?"#4A90D9":"#0d1f35", border:`1px solid ${active===h?"#4A90D9":"#1a3050"}`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:11, color:active===h?"#fff":"#4a6a8a", cursor:"pointer", transition:"all 0.2s" }}>
            H{h+1}
          </div>
        ))}
      </div>
      {active!==null && <div style={{ textAlign:"center", fontSize:12, color:"#4A90D9" }}>Head {active+1} attending to {DESCS[active]} features</div>}
      <div style={{ textAlign:"center", marginTop:"1.5rem" }}>
        <div style={{ display:"inline-block", padding:"10px 32px", background:"linear-gradient(135deg,#1ABC9C22,#4A90D922)", border:"1px solid #1ABC9C44", borderRadius:4, fontSize:13, color:"#1ABC9C", letterSpacing:1 }}>→ Fused Emotion Representation</div>
      </div>
    </div>
  );
}

/* ── Emotion result bars ── */
function EmotionBars({ results, label }) {
  if (!results) return null;
  const top = results[0];
  return (
    <div style={{ animation:"fadeIn 0.4s both", width:"100%" }}>
      <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:"1.5rem", flexWrap:"wrap" }}>
        <div style={{ width:50, height:50, borderRadius:"50%", background:`${top.color}22`, border:`2px solid ${top.color}`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:20, flexShrink:0 }}>◎</div>
        <div style={{ flex:1, minWidth:100 }}>
          <div style={{ fontFamily:"'Space Mono',monospace", fontSize:10, color:"#8899aa", letterSpacing:2, textTransform:"uppercase" }}>Primary emotion</div>
          <div style={{ fontWeight:800, fontSize:20, color:top.color }}>{top.label}</div>
        </div>
        <div style={{ fontFamily:"'Space Mono',monospace", fontSize:26, fontWeight:700, color:top.color }}>{top.score}%</div>
      </div>
      {results.map(e => (
        <div key={e.label} style={{ marginBottom:10 }}>
          <div style={{ display:"flex", justifyContent:"space-between", fontFamily:"'Space Mono',monospace", fontSize:11, color:"#8899aa", marginBottom:4 }}>
            <span>{e.label}</span><span style={{ color:e.color }}>{e.score}%</span>
          </div>
          <div style={{ height:5, background:"#0d2040", borderRadius:3, overflow:"hidden" }}>
            <div style={{ height:"100%", width:`${e.score}%`, background:e.color, borderRadius:3, transition:"width 0.8s ease" }} />
          </div>
        </div>
      ))}
      <div style={{ marginTop:"1rem", fontFamily:"'Space Mono',monospace", fontSize:10, color:"#3a5a7a", borderTop:"1px solid #0d2040", paddingTop:"0.75rem" }}>
        ↳ Analyzed via {label} · Attention Bottleneck v1.0
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════
   UPLOAD ANALYZER
══════════════════════════════════════════════════ */
function UploadAnalyzer() {
  const [tab, setTab]         = useState("image");
  const [dragging, setDrag]   = useState(false);
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [text, setText]       = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const inputRef = useRef(null);

  const TABS = [
    { id:"image", label:"📷  Image", accept:"image/*" },
    { id:"video", label:"🎬  Video", accept:"video/*" },
    { id:"audio", label:"🎙  Audio", accept:"audio/*" },
    { id:"text",  label:"✦  Text",  accept:null },
  ];

  const handleFile = (f) => {
    if (!f) return;
    setFile(f); setResults(null);
    if (f.type.startsWith("image/") || f.type.startsWith("video/")) setPreview(URL.createObjectURL(f));
    else setPreview(null);
  };

  const onDrop = (e) => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]); };

  const analyze = async () => {
    setLoading(true); setResults(null);
    await new Promise(r => setTimeout(r, 1600 + Math.random() * 700));
    setResults(fakeAnalyze()); setLoading(false);
  };

  const canAnalyze = tab === "text" ? text.trim().length > 3 : !!file;

  return (
    <div>
      {/* Tabs */}
      <div style={{ display:"flex", gap:8, marginBottom:"2rem", flexWrap:"wrap" }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => { setTab(t.id); setFile(null); setPreview(null); setResults(null); setText(""); }}
            style={{ padding:"10px 22px", borderRadius:4, border:`1px solid ${tab===t.id?"#4A90D9":"#0d2040"}`, background:tab===t.id?"#4A90D920":"#0a1828", color:tab===t.id?"#4A90D9":"#8899aa", fontFamily:"'Space Mono',monospace", fontSize:12, cursor:"pointer", transition:"all 0.2s", letterSpacing:1 }}>
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"minmax(0,1fr) minmax(0,1fr)", gap:"1.5rem" }}>
        {/* Input */}
        <div>
          {tab === "text" ? (
            <textarea value={text} onChange={e => { setText(e.target.value); setResults(null); }}
              placeholder="Paste or type any text — a sentence, a paragraph, a chat snippet..."
              style={{ width:"100%", minHeight:220, background:"#0a1828", border:"1px solid #0d2040", borderRadius:8, padding:"1.25rem", color:"#c8d8e8", fontFamily:"'Space Mono',monospace", fontSize:13, lineHeight:1.8, resize:"vertical", outline:"none", boxSizing:"border-box" }} />
          ) : (
            <div onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop} onClick={() => inputRef.current?.click()}
              style={{ minHeight:220, background:dragging?"#0d2a4a":"#0a1828", border:`2px dashed ${dragging?"#4A90D9":"#1a3050"}`, borderRadius:8, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:"1rem", cursor:"pointer", transition:"all 0.2s", padding:"2rem", overflow:"hidden", position:"relative" }}>
              <input ref={inputRef} type="file" accept={TABS.find(t => t.id===tab)?.accept} style={{ display:"none" }} onChange={e => handleFile(e.target.files[0])} />
              {preview && tab==="image" ? (
                <img src={preview} alt="" style={{ maxWidth:"100%", maxHeight:175, borderRadius:6, objectFit:"contain" }} />
              ) : preview && tab==="video" ? (
                <video src={preview} controls style={{ maxWidth:"100%", maxHeight:175, borderRadius:6 }} />
              ) : (
                <>
                  <div style={{ fontSize:38 }}>{ {image:"🖼",video:"🎬",audio:"🎵"}[tab] }</div>
                  <div style={{ fontFamily:"'Space Mono',monospace", fontSize:12, color:"#8899aa", textAlign:"center", lineHeight:1.8 }}>Drop a {tab} file here<br/>or click to browse</div>
                </>
              )}
              {file && <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color:"#4A90D9", marginTop:4 }}>✓ {file.name}</div>}
            </div>
          )}

          <button onClick={analyze} disabled={!canAnalyze || loading}
            style={{ marginTop:"1rem", width:"100%", padding:"14px", background:canAnalyze&&!loading?"#0d2a4a":"#0a1828", border:`1px solid ${canAnalyze&&!loading?"#4A90D9":"#0d2040"}`, borderRadius:6, color:canAnalyze&&!loading?"#4A90D9":"#3a5a7a", fontFamily:"'Space Mono',monospace", fontSize:13, letterSpacing:2, cursor:canAnalyze&&!loading?"pointer":"not-allowed", transition:"all 0.3s" }}>
            {loading ? "◌  Analyzing..." : "▶  Run emotion analysis"}
          </button>

          {loading && (
            <div style={{ marginTop:"1rem", fontFamily:"'Space Mono',monospace", fontSize:11, color:"#8899aa", lineHeight:2.2 }}>
              {["Extracting features…","Running attention bottleneck…","Classifying emotions…"].map((s, i) => (
                <div key={i} style={{ animation:`fadeIn 0.3s ${i*0.45}s both`, display:"flex", alignItems:"center", gap:8 }}>
                  <div style={{ width:6, height:6, borderRadius:"50%", background:"#4A90D9", animation:"pulse 1s infinite" }} />{s}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Results */}
        <div style={{ background:"#0a1828", border:"1px solid #0d2040", borderRadius:8, padding:"1.5rem", minHeight:220, display:"flex", alignItems:results||loading?"flex-start":"center", justifyContent:results||loading?"flex-start":"center" }}>
          {!results && !loading && (
            <div style={{ textAlign:"center", color:"#3a5a7a", fontFamily:"'Space Mono',monospace", fontSize:12 }}>
              <div style={{ fontSize:32, marginBottom:"1rem" }}>◎</div>
              Predictions appear here
            </div>
          )}
          {loading && (
            <div style={{ width:"100%" }}>
              {[70,45,58,32,42,28].map((w, i) => (
                <div key={i} style={{ marginBottom:16 }}>
                  <div style={{ height:10, background:"#0d2040", borderRadius:3, marginBottom:6, width:`${w}%`, animation:"pulse 1.2s infinite" }} />
                  <div style={{ height:5, background:"#0d2040", borderRadius:3 }} />
                </div>
              ))}
            </div>
          )}
          {results && <EmotionBars results={results} label={{ image:"image · visual stream", video:"video · visual + audio", audio:"audio · prosody", text:"text · semantic" }[tab]} />}
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════
   LIVE CAMERA
══════════════════════════════════════════════════ */
function LiveCamera() {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const streamRef   = useRef(null);
  const timerRef    = useRef(null);

  const [camOn,    setCamOn]    = useState(false);
  const [results,  setResults]  = useState(null);
  const [snap,     setSnap]     = useState(null);
  const [error,    setError]    = useState(null);
  const [scanning, setScanning] = useState(false);
  const [mode,     setMode]     = useState("photo");

  const stopScan = () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } setScanning(false); };

  const startScan = () => {
    setScanning(true);
    timerRef.current = setInterval(() => setResults(fakeAnalyze()), 1500);
  };

  const openCam = async () => {
    try {
      setError(null);
      const s = await navigator.mediaDevices.getUserMedia({ video:{ facingMode:"user", width:640, height:480 }, audio:false });
      streamRef.current = s;
      if (videoRef.current) { videoRef.current.srcObject = s; await videoRef.current.play(); }
      setCamOn(true);
      if (mode === "live") startScan();
    } catch {
      setError("Camera access denied. Allow camera permissions and try again.");
    }
  };

  const closeCam = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    stopScan(); setCamOn(false); setResults(null); setSnap(null);
  };

  const capture = () => {
    const v = videoRef.current; const c = canvasRef.current; if (!v || !c) return;
    c.width = v.videoWidth || 640; c.height = v.videoHeight || 480;
    c.getContext("2d").drawImage(v, 0, 0);
    setSnap(c.toDataURL("image/jpeg", 0.85)); setResults(null);
    setTimeout(() => setResults(fakeAnalyze()), 900);
  };

  const switchMode = (m) => {
    setMode(m); stopScan(); setResults(null); setSnap(null);
    if (camOn && m === "live") startScan();
  };

  useEffect(() => () => { streamRef.current?.getTracks().forEach(t => t.stop()); stopScan(); }, []);

  const top = results?.[0];

  return (
    <div style={{ display:"grid", gridTemplateColumns:"minmax(0,1fr) minmax(0,320px)", gap:"1.5rem", alignItems:"start" }}>
      {/* Camera */}
      <div style={{ background:"#050d1a", border:"1px solid #0d2040", borderRadius:12, overflow:"hidden" }}>
        {/* Mode toggle */}
        <div style={{ display:"flex", borderBottom:"1px solid #0d2040" }}>
          {["photo","live"].map(m => (
            <button key={m} onClick={() => switchMode(m)}
              style={{ flex:1, padding:"12px", background:mode===m?"#0a1828":"transparent", border:"none", borderBottom:mode===m?"2px solid #4A90D9":"2px solid transparent", color:mode===m?"#4A90D9":"#8899aa", fontFamily:"'Space Mono',monospace", fontSize:11, letterSpacing:2, cursor:"pointer", transition:"all 0.2s", textTransform:"uppercase" }}>
              {m==="photo"?"📷  Snapshot":"◉  Live scan"}
            </button>
          ))}
        </div>

        {/* Viewfinder */}
        <div style={{ position:"relative", background:"#000", aspectRatio:"4/3", display:"flex", alignItems:"center", justifyContent:"center", overflow:"hidden" }}>
          <video ref={videoRef} muted playsInline style={{ width:"100%", height:"100%", objectFit:"cover", display:camOn?"block":"none", transform:"scaleX(-1)" }} />
          <canvas ref={canvasRef} style={{ display:"none" }} />

          {!camOn && (
            <div style={{ textAlign:"center", color:"#3a5a7a", fontFamily:"'Space Mono',monospace" }}>
              <div style={{ fontSize:44, marginBottom:"1rem" }}>📷</div>
              <div style={{ fontSize:12 }}>Camera is off</div>
            </div>
          )}

          {/* Live overlay */}
          {camOn && mode==="live" && scanning && (
            <div style={{ position:"absolute", inset:0, pointerEvents:"none" }}>
              <div style={{ position:"absolute", top:14, left:14, width:26, height:26, borderTop:"2px solid #4A90D9", borderLeft:"2px solid #4A90D9" }} />
              <div style={{ position:"absolute", top:14, right:14, width:26, height:26, borderTop:"2px solid #4A90D9", borderRight:"2px solid #4A90D9" }} />
              <div style={{ position:"absolute", bottom:14, left:14, width:26, height:26, borderBottom:"2px solid #4A90D9", borderLeft:"2px solid #4A90D9" }} />
              <div style={{ position:"absolute", bottom:14, right:14, width:26, height:26, borderBottom:"2px solid #4A90D9", borderRight:"2px solid #4A90D9" }} />
              <div style={{ position:"absolute", bottom:10, left:"50%", transform:"translateX(-50%)", fontFamily:"'Space Mono',monospace", fontSize:10, color:"#4A90D9", letterSpacing:2, background:"#050d1a99", padding:"3px 10px", borderRadius:2 }}>◉ SCANNING</div>
              {top && (
                <div style={{ position:"absolute", top:10, left:"50%", transform:"translateX(-50%)", background:`${top.color}dd`, padding:"4px 16px", borderRadius:20, fontFamily:"'Space Mono',monospace", fontSize:12, color:"#fff", fontWeight:700, whiteSpace:"nowrap" }}>
                  {top.label}  {top.score}%
                </div>
              )}
            </div>
          )}

          {snap && mode==="photo" && (
            <img src={snap} alt="" style={{ position:"absolute", inset:0, width:"100%", height:"100%", objectFit:"cover", transform:"scaleX(-1)" }} />
          )}
        </div>

        {/* Controls */}
        <div style={{ padding:"1rem", display:"flex", gap:"0.75rem", background:"#070f1e" }}>
          {!camOn ? (
            <button onClick={openCam} style={{ flex:1, padding:"12px", background:"#1A3A6A", border:"1px solid #4A90D9", borderRadius:6, color:"#4A90D9", fontFamily:"'Space Mono',monospace", fontSize:12, letterSpacing:2, cursor:"pointer" }}>
              ▶  Start camera
            </button>
          ) : (
            <>
              {mode==="photo" && (
                <button onClick={capture} style={{ flex:2, padding:"12px", background:"#1A3A6A", border:"1px solid #4A90D9", borderRadius:6, color:"#4A90D9", fontFamily:"'Space Mono',monospace", fontSize:12, letterSpacing:2, cursor:"pointer" }}>
                  📷  Capture &amp; Analyze
                </button>
              )}
              {mode==="live" && !scanning && (
                <button onClick={startScan} style={{ flex:2, padding:"12px", background:"#1A3A6A", border:"1px solid #1ABC9C", borderRadius:6, color:"#1ABC9C", fontFamily:"'Space Mono',monospace", fontSize:12, letterSpacing:2, cursor:"pointer" }}>
                  ◉  Start live scan
                </button>
              )}
              <button onClick={closeCam} style={{ flex:1, padding:"12px", background:"#1a0a0a", border:"1px solid #E84040", borderRadius:6, color:"#E84040", fontFamily:"'Space Mono',monospace", fontSize:12, letterSpacing:2, cursor:"pointer" }}>
                ■  Stop
              </button>
            </>
          )}
        </div>
        {error && <div style={{ padding:"0.75rem 1rem", fontFamily:"'Space Mono',monospace", fontSize:11, color:"#E84040", background:"#1a0a0a", borderTop:"1px solid #2a1010" }}>{error}</div>}
      </div>

      {/* Results */}
      <div style={{ background:"#0a1828", border:"1px solid #0d2040", borderRadius:12, padding:"1.5rem", minHeight:280, display:"flex", flexDirection:"column" }}>
        <div style={{ fontFamily:"'Space Mono',monospace", fontSize:10, color:"#8899aa", letterSpacing:2, textTransform:"uppercase", marginBottom:"1.5rem" }}>
          {mode==="live"?"◉ Real-time output":"Snapshot analysis"}
        </div>
        {!results && (
          <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", color:"#3a5a7a", fontFamily:"'Space Mono',monospace", fontSize:12, textAlign:"center", gap:"0.75rem" }}>
            <div style={{ fontSize:32 }}>◎</div>
            {!camOn ? "Start the camera to begin" : mode==="photo" ? "Capture a frame to analyze" : "Starting live scan…"}
          </div>
        )}
        {results && <EmotionBars results={results} label={mode==="live"?"live camera · real-time":"snapshot · visual"} />}
      </div>
    </div>
  );
}

/* ── Waveform bar ── */
function WaveBar({ delay }) {
  return <div style={{ width:3, background:"#4A90D9", borderRadius:2, animation:`wave 1.2s ease-in-out ${delay}s infinite alternate` }} />;
}

/* ══════════════════════════════════════════════════
   MAIN
══════════════════════════════════════════════════ */
export default function MainPage() {
  const [scrollY, setScrollY] = useState(0);
  useEffect(() => {
    const fn = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", fn);
    return () => window.removeEventListener("scroll", fn);
  }, []);

  const css = `
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#050d1a;color:#c8d8e8}
    @keyframes wave{from{height:6px;opacity:.4}to{height:28px;opacity:1}}
    @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
    @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
    @keyframes spin{to{transform:rotate(360deg)}}
    @keyframes pulse{0%,100%{opacity:.5}50%{opacity:1}}
    .card:hover{transform:translateY(-4px);border-color:#2a4a7a!important}
    .card{transition:all .3s}
    .emotion-dot{transition:transform .2s;cursor:default}
    .emotion-dot:hover{transform:scale(1.3)}
    textarea:focus{outline:none;border-color:#4A90D9!important}
    @media(max-width:640px){
      .two-col{grid-template-columns:1fr!important}
      .cam-grid{grid-template-columns:1fr!important}
    }
  `;

  const Sec = ({ num, color, children }) => (
    <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color, letterSpacing:3, marginBottom:"1rem", textTransform:"uppercase" }}>§ {num} — {children}</div>
  );
  const H2 = ({ children }) => (
    <h2 style={{ fontSize:"clamp(1.5rem,3vw,2.2rem)", fontWeight:700, color:"#e8f0ff", marginBottom:"3rem", lineHeight:1.2 }}>{children}</h2>
  );

  return (
    <>
      <style>{css}</style>
      <div style={{ fontFamily:"'Syne',sans-serif", background:"#050d1a", minHeight:"100vh", color:"#c8d8e8" }}>

        {/* Nav */}
        <nav style={{ position:"fixed", top:0, left:0, right:0, zIndex:100, background:scrollY>40?"rgba(5,13,26,.95)":"transparent", backdropFilter:scrollY>40?"blur(12px)":"none", borderBottom:scrollY>40?"1px solid #0d2040":"none", transition:"all .3s", padding:"1rem 2rem", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
          <div style={{ fontFamily:"'Space Mono',monospace", fontSize:13, color:"#4A90D9", letterSpacing:2 }}>MERS<span style={{ color:"#1ABC9C" }}>·</span>AB</div>
          <div style={{ display:"flex", gap:"1.5rem", flexWrap:"wrap" }}>
            {["Abstract","Architecture","Demo","Camera","Results","Team"].map(l => (
              <a key={l} href={`#${l.toLowerCase()}`} style={{ color:"#8899aa", fontSize:12, letterSpacing:1, textDecoration:"none", fontFamily:"'Space Mono',monospace", transition:"color .2s" }}
                onMouseEnter={e => e.target.style.color="#c8d8e8"} onMouseLeave={e => e.target.style.color="#8899aa"}>{l}</a>
            ))}
          </div>
        </nav>

        {/* Hero */}
        <section style={{ position:"relative", minHeight:"100vh", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:"6rem 2rem 4rem", overflow:"hidden", background:"radial-gradient(ellipse at 50% 60%,#0a1f40 0%,#050d1a 70%)" }}>
          <ParticleField />
          <div style={{ position:"absolute", width:500, height:500, border:"1px solid #0d2040", borderRadius:"50%", top:"50%", left:"50%", transform:"translate(-50%,-50%)", animation:"spin 40s linear infinite" }} />
          <div style={{ position:"absolute", width:360, height:360, border:"1px dashed #0d3060", borderRadius:"50%", top:"50%", left:"50%", transform:"translate(-50%,-50%)", animation:"spin 25s linear infinite reverse" }} />
          <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, letterSpacing:3, color:"#4A90D9", border:"1px solid #1a3a6a", padding:"6px 18px", borderRadius:2, marginBottom:"2rem", position:"relative", zIndex:1 }}>
            Bachelor's Thesis · 2026
          </div>
          <h1 style={{ fontSize:"clamp(2rem,5vw,3.8rem)", fontWeight:800, lineHeight:1.1, textAlign:"center", maxWidth:800, position:"relative", zIndex:1, marginBottom:"1.5rem" }}>
            <span style={{ color:"#e8f0ff" }}>Development of a </span>
            <span style={{ backgroundImage:"linear-gradient(135deg,#4A90D9,#1ABC9C)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>Multimodal Emotion</span>
            <span style={{ color:"#e8f0ff" }}> Recognition System</span>
          </h1>
          <p style={{ fontFamily:"'Space Mono',monospace", fontSize:13, color:"#8899aa", textAlign:"center", maxWidth:480, lineHeight:1.8, position:"relative", zIndex:1, marginBottom:"3rem" }}>
            Using Attention Bottleneck Mechanism for cross-modal fusion of audio, visual &amp; textual streams
          </p>
          <div style={{ display:"flex", gap:4, alignItems:"center", height:40, position:"relative", zIndex:1, marginBottom:"3rem" }}>
            {[...Array(24)].map((_, i) => <WaveBar key={i} delay={(i*0.05)%1.2} />)}
          </div>
          <div style={{ display:"flex", gap:"3rem", position:"relative", zIndex:1, flexWrap:"wrap", justifyContent:"center" }}>
            {[{l:"Accuracy",v:84,s:"%"},{l:"Modalities",v:3,s:""},{l:"Emotion classes",v:6,s:""},{l:"Attn heads",v:8,s:""}].map(({l,v,s}) => (
              <div key={l} style={{ textAlign:"center" }}>
                <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"clamp(1.8rem,4vw,2.8rem)", fontWeight:700, color:"#4A90D9", lineHeight:1 }}><Counter target={v} suffix={s} /></div>
                <div style={{ fontSize:12, color:"#8899aa", letterSpacing:2, marginTop:6, textTransform:"uppercase" }}>{l}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Abstract */}
        <section id="abstract" style={{ padding:"6rem 2rem", maxWidth:900, margin:"0 auto" }}>
          <Sec num="01" color="#4A90D9">Abstract</Sec>
          <H2>Why single-modal approaches fall short</H2>
          <p style={{ color:"#8899aa", lineHeight:2, fontSize:15, marginBottom:"1.5rem" }}>
            Human emotion is inherently multimodal — a subtle smile, a trembling voice, and carefully chosen words each carry independent signals that, when fused intelligently, reveal far richer affective states than any single channel can provide alone.
          </p>
          <p style={{ color:"#8899aa", lineHeight:2, fontSize:15 }}>
            This thesis proposes a novel architecture built around an <span style={{ color:"#4A90D9", fontFamily:"'Space Mono',monospace", fontSize:13 }}>attention bottleneck transformer</span> that compresses inter-modal information through a small set of shared latent tokens — reducing cost while preserving cross-modal expressivity.
          </p>
          <div style={{ display:"flex", gap:"1.5rem", marginTop:"3rem", flexWrap:"wrap" }}>
            {MODALITIES.map(m => (
              <div key={m.name} className="card" style={{ flex:"1 1 180px", padding:"1.5rem", background:"#0a1828", border:"1px solid #0d2040", borderRadius:8 }}>
                <div style={{ fontSize:24, marginBottom:"0.75rem" }}>{m.icon}</div>
                <div style={{ fontWeight:700, color:m.color, marginBottom:"0.5rem", fontSize:15 }}>{m.name}</div>
                <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color:"#8899aa", lineHeight:1.8 }}>{m.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Architecture */}
        <section id="architecture" style={{ padding:"6rem 2rem", background:"#070f1e" }}>
          <div style={{ maxWidth:900, margin:"0 auto" }}>
            <Sec num="02" color="#1ABC9C">Architecture</Sec>
            <H2>The attention bottleneck pipeline</H2>
            <div style={{ position:"relative" }}>
              <div style={{ position:"absolute", left:20, top:0, bottom:0, width:1, background:"linear-gradient(to bottom,#4A90D9,#1ABC9C)" }} />
              <div style={{ display:"flex", flexDirection:"column", gap:"2rem", paddingLeft:"3.5rem" }}>
                {TIMELINE.map((t, i) => (
                  <div key={i} style={{ position:"relative" }}>
                    <div style={{ position:"absolute", left:-46, width:22, height:22, borderRadius:"50%", background:"#050d1a", border:`2px solid ${["#4A90D9","#E84040","#1ABC9C","#F4C430"][i]}`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:10, color:["#4A90D9","#E84040","#1ABC9C","#F4C430"][i], marginTop:2 }}>{i+1}</div>
                    <div className="card" style={{ padding:"1.25rem 1.5rem", background:"#0a1828", border:"1px solid #0d2040", borderRadius:8 }}>
                      <div style={{ display:"flex", alignItems:"center", gap:"0.75rem", marginBottom:"0.5rem" }}>
                        <span style={{ fontSize:18 }}>{t.icon}</span>
                        <span style={{ fontWeight:700, color:"#e8f0ff", fontSize:15 }}>{t.phase}</span>
                      </div>
                      <div style={{ fontFamily:"'Space Mono',monospace", fontSize:12, color:"#8899aa" }}>{t.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ marginTop:"4rem", padding:"2rem", background:"#0a1828", border:"1px solid #0d2040", borderRadius:8 }}>
              <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color:"#8899aa", letterSpacing:2, marginBottom:"0.5rem", textTransform:"uppercase" }}>Interactive — hover over attention heads</div>
              <AttentionViz />
            </div>
          </div>
        </section>

        {/* Emotion map */}
        <section style={{ padding:"6rem 2rem", maxWidth:900, margin:"0 auto" }}>
          <Sec num="03" color="#E84040">Output space</Sec>
          <H2>Six-class emotion taxonomy</H2>
          <div style={{ position:"relative", paddingTop:"60%", background:"#0a1828", border:"1px solid #0d2040", borderRadius:8, overflow:"hidden" }}>
            <div style={{ position:"absolute", inset:0 }}>
              {EMOTIONS.map((e, i) => (
                <div key={i} className="emotion-dot" style={{ position:"absolute", left:`${e.x}%`, top:`${e.y}%`, transform:"translate(-50%,-50%)", textAlign:"center" }}>
                  <div style={{ width:56, height:56, borderRadius:"50%", background:`${e.color}22`, border:`2px solid ${e.color}66`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:22, color:e.color, animation:`float ${3+i*.4}s ease-in-out ${i*.3}s infinite`, margin:"0 auto" }}>◎</div>
                  <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color:e.color, marginTop:6, letterSpacing:1 }}>{e.label}</div>
                </div>
              ))}
              <svg style={{ position:"absolute", inset:0, width:"100%", height:"100%", pointerEvents:"none" }} viewBox="0 0 100 100" preserveAspectRatio="none">
                {EMOTIONS.map((e, i) => EMOTIONS.slice(i+1).map((f, j) => <line key={`${i}-${j}`} x1={e.x} y1={e.y} x2={f.x} y2={f.y} stroke="#1a3a5a" strokeWidth="0.3" />))}
              </svg>
            </div>
          </div>
        </section>

        {/* ════ DEMO — Upload ════ */}
        <section id="demo" style={{ padding:"6rem 2rem", background:"#070f1e" }}>
          <div style={{ maxWidth:960, margin:"0 auto" }}>
            <Sec num="04" color="#F4C430">Live demo</Sec>
            <H2>Upload media to analyze emotions</H2>
            <p style={{ color:"#8899aa", fontSize:14, lineHeight:1.9, marginBottom:"2.5rem", fontFamily:"'Space Mono',monospace" }}>
              Drop an image, video clip, audio file, or paste text. The attention bottleneck model extracts features across all available modalities and outputs per-class confidence scores.
            </p>
            <UploadAnalyzer />
          </div>
        </section>

        {/* ════ DEMO — Camera ════ */}
        <section id="camera" style={{ padding:"6rem 2rem", maxWidth:960, margin:"0 auto" }}>
          <Sec num="05" color="#1ABC9C">Real-time detection</Sec>
          <H2>Live camera emotion scanner</H2>
          <p style={{ color:"#8899aa", fontSize:14, lineHeight:1.9, marginBottom:"2.5rem", fontFamily:"'Space Mono',monospace" }}>
            Activate your webcam and analyze facial expressions live. <span style={{ color:"#4A90D9" }}>Snapshot</span> mode captures one frame; <span style={{ color:"#E84040" }}>Live scan</span> streams continuous predictions every 1.5 s with a real-time overlay.
          </p>
          <LiveCamera />
        </section>

        {/* Results */}
        <section id="results" style={{ padding:"6rem 2rem", background:"#070f1e" }}>
          <div style={{ maxWidth:900, margin:"0 auto" }}>
            <Sec num="06" color="#F4C430">Results</Sec>
            <H2>Benchmark comparisons</H2>
            <div style={{ overflowX:"auto" }}>
              <table style={{ width:"100%", borderCollapse:"collapse", fontFamily:"'Space Mono',monospace", fontSize:13 }}>
                <thead>
                  <tr style={{ borderBottom:"1px solid #0d2040" }}>
                    {["Model","Audio only","Visual only","Text only","Multimodal (ours)"].map(h => (
                      <th key={h} style={{ padding:"1rem 1.5rem", color:"#8899aa", textAlign:"left", fontSize:11, letterSpacing:1, textTransform:"uppercase" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[["Baseline LSTM","68%","65%","71%","76%"],["Transformer (vanilla)","72%","69%","75%","80%"],["Ours (ABM)","74%","71%","77%","84%"]].map(([name,...vals], i) => (
                    <tr key={i} style={{ borderBottom:"1px solid #0d2040", background:i===2?"#0d1f35":"transparent" }}>
                      <td style={{ padding:"1rem 1.5rem", color:i===2?"#1ABC9C":"#c8d8e8", fontWeight:i===2?700:400 }}>
                        {name}{i===2&&<span style={{ fontSize:10, color:"#1ABC9C", marginLeft:8, background:"#1ABC9C22", padding:"2px 8px", borderRadius:2 }}>★ ours</span>}
                      </td>
                      {vals.map((v, j) => <td key={j} style={{ padding:"1rem 1.5rem", color:i===2&&j===3?"#4A90D9":"#8899aa", fontWeight:i===2&&j===3?700:400, fontSize:i===2&&j===3?15:13 }}>{v}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop:"3rem", display:"flex", gap:"1.5rem", flexWrap:"wrap" }}>
              {[{label:"Anger",acc:88,color:"#E84040"},{label:"Joy",acc:91,color:"#F4C430"},{label:"Sadness",acc:85,color:"#4A90D9"},{label:"Fear",acc:79,color:"#9B59B6"},{label:"Surprise",acc:82,color:"#1ABC9C"},{label:"Disgust",acc:76,color:"#E67E22"}].map(({label,acc,color}) => (
                <div key={label} style={{ flex:"1 1 110px", minWidth:90 }}>
                  <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color:"#8899aa", marginBottom:8 }}>{label}</div>
                  <div style={{ height:6, background:"#0d2040", borderRadius:3, overflow:"hidden" }}>
                    <div style={{ height:"100%", width:`${acc}%`, background:color, borderRadius:3 }} />
                  </div>
                  <div style={{ fontFamily:"'Space Mono',monospace", fontSize:12, color, marginTop:6 }}>{acc}%</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Team */}
        <section id="team" style={{ padding:"6rem 2rem", maxWidth:900, margin:"0 auto" }}>
          <Sec num="07" color="#9B59B6">Authors</Sec>
          <H2>Research team</H2>
          <div style={{ display:"flex", gap:"1.5rem", flexWrap:"wrap" }}>
            {[{initials:"SN",name:"Student Name",role:"Author",color:"#4A90D9"},{initials:"SP",name:"Supervisor Name",role:"Scientific Advisor",color:"#1ABC9C"}].map(({initials,name,role,color}) => (
              <div key={initials} className="card" style={{ flex:"1 1 220px", padding:"1.75rem", background:"#0a1828", border:"1px solid #0d2040", borderRadius:8, display:"flex", gap:"1rem", alignItems:"center" }}>
                <div style={{ width:52, height:52, borderRadius:"50%", background:`${color}22`, border:`2px solid ${color}55`, display:"flex", alignItems:"center", justifyContent:"center", fontFamily:"'Space Mono',monospace", fontWeight:700, fontSize:14, color, flexShrink:0 }}>{initials}</div>
                <div>
                  <div style={{ fontWeight:700, color:"#e8f0ff", marginBottom:4 }}>{name}</div>
                  <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color:"#8899aa" }}>{role}</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer style={{ padding:"2rem", borderTop:"1px solid #0d2040", display:"flex", justifyContent:"space-between", alignItems:"center", flexWrap:"wrap", gap:"1rem" }}>
          <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color:"#3a5a7a" }}>Development of a Multimodal Emotion Recognition System using Attention Bottleneck Mechanism</div>
          <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color:"#3a5a7a" }}>© 2025</div>
        </footer>
      </div>
    </>
  );
}
