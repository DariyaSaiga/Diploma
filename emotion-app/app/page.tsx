import Navbar from "@/components/Navbar";
import FAQ from "@/components/FAQ";

// ── Data ──────────────────────────────────────────────────────────────────────

const STATS = [
  { value: "6", label: "Emotion classes" },
  { value: "3", label: "Modalities (text, audio, video)" },
  { value: "87%", label: "Macro-F1 on CMU-MOSEI" },
  { value: "13,934", label: "Training samples" },
];

const EVENTS = [
  {
    name: "Text Modality",
    accent: "BERT Encoder",
    desc: "Frozen bert-base-uncased extracts the mean of the last 4 hidden layers into a [B, 50, 768] tensor, providing rich contextual language representations without fine-tuning.",
    date: "Implemented",
  },
  {
    name: "Audio Modality",
    accent: "COVAREP Features",
    desc: "74-dimensional acoustic low-level descriptors per frame, fixed to 60 frames per sample. Processed via Conv1D projection and positional encoding into a [B, 60, 128] tensor.",
    date: "Implemented",
  },
  {
    name: "Visual Modality",
    accent: "OpenFace AU",
    desc: "35-dimensional facial action unit features per frame extracted by OpenFace, fixed to 60 frames. Projected with Conv1D and positional encoding into a [B, 60, 128] tensor.",
    date: "Implemented",
  },
  {
    name: "Bottleneck Fusion",
    accent: "Cross-modal Attention",
    desc: "16 learnable bottleneck tokens act as a compact communication channel across all three modalities over 2 attention layers, enabling controlled cross-modal interaction without full pairwise attention.",
    date: "Core module",
  },
];

const REVIEWS = [
  {
    role: "Dataset — CMU-MOSEI",
    text: "Large-scale multimodal benchmark with 13,934 training / 1,569 validation / 3,957 test samples across 6 emotion categories: happiness, sadness, anger, surprise, disgust, and fear.",
  },
  {
    role: "Training — AdamW + Cosine Annealing",
    text: "Batch size 8, learning rate 1e-4, gradient clipping at 1.0, early stopping on validation Macro-F1 with patience 10. Weighted BCEWithLogitsLoss handles severe class imbalance (fear: 9.6%, happiness: 62.7%).",
  },
  {
    role: "Evaluation — Macro-F1 & Avg. WA",
    text: "Primary metric is macro-averaged F1 across all 6 classes. Per-class threshold tuning on validation set further improves minority-class detection, especially for fear and surprise.",
  },
];

const PLANS = [
  {
    tier: "Late Fusion Baseline",
    price: "Baseline",
    period: "Independent unimodal classifiers combined at decision level",
    featured: false,
    features: [
      "Separate text, audio, vision heads",
      "Mean-pooled unimodal representations",
      "Simple concatenation before classifier",
      "No cross-modal information exchange",
      "Lower Macro-F1 on minority classes",
    ],
  },
  {
    tier: "Bottleneck Fusion Model",
    price: "Proposed",
    period: "Cross-modal attention via shared learnable bottleneck tokens",
    featured: true,
    features: [
      "Frozen BERT + Conv1D audio & visual encoders",
      "16 learnable bottleneck tokens (hidden dim 128)",
      "2-layer cross-modal attention fusion",
      "Auxiliary unimodal classification heads",
      "Modality dropout (p=0.05) for robustness",
      "Per-class threshold tuning at inference",
    ],
  },
  {
    tier: "Ablation Variants",
    price: "Analysis",
    period: "Systematic component-wise evaluation",
    featured: false,
    features: [
      "Single BERT layer vs. mean of last 4 layers",
      "Modality dropout at p=0, 0.05, 0.10, 0.15",
      "With vs. without auxiliary losses",
      "Default threshold 0.5 vs. tuned thresholds",
      "Bottleneck vs. late fusion comparison",
    ],
  },
];

// ── Page ──────────────────────────────────────────────────────────────────────

export default function Home() {
  return (
    <>
      {/* ── NAVBAR ── */}
      <Navbar />

      <main>

        {/* ── HERO ── */}
        <section className="hero">
          <div className="hero-media">
            {/*
              Добавьте ваше фото:
              import Image from "next/image";
              <Image src="/hero.jpg" alt="MultiMOOD" fill style={{ objectFit: "cover" }} priority />
            */}
            <img
              src="/faces.jpg"
              alt="Your photo / video"
              className="img-placeholder"
            />
            <div className="hero-overlay">
              <h1 className="hero-title">
                Understanding<br />Human Emotions
              </h1>
              <p className="hero-sub">
                A multimodal AI system that recognises emotions from video and audio
                using an Attention Bottleneck Mechanism — more accurately than any single-modality approach.
              </p>
              <div className="hero-btns">
                <a href="#about" className="btn btn-white">Explore the System →</a>
                <a href="#about" className="btn btn-outline-white">Learn More</a>
              </div>
            </div>
          </div>
        </section>

        {/* ── STATS ── */}
        <section className="stats">
          {STATS.map((s) => (
            <div key={s.label} className="stat">
              <div className="stat-value">{s.value}</div>
              <div className="stat-label">{s.label}</div>
            </div>
          ))}
        </section>

        {/* ── ABOUT ── */}
        <section className="about" id="about">
          <div className="about-left">
            <h2 className="section-title">
              About the<br />
              <span className="italic">MultiMOOD</span>
            </h2>

            <div className="about-card">
              <div className="about-card-tag">
                <span className="about-card-dot" />
                Accuracy &amp; Speed
              </div>
              <div className="about-card-body">
                <p>
                  Clean architecture with no redundant dependencies. The system
                  processes video input and returns emotion predictions with minimal latency
                  through an efficient feature extraction pipeline.
                </p>
              </div>
              <div className="about-card-img">
                {/* <Image src="/interface.jpg" alt="Interface" fill style={{ objectFit: "cover" }} /> */}
                Your interface screenshot
              </div>
            </div>

            <div className="about-media-grid">
              <div className="about-media-item" style={{ background: "#f0f0f0" }}>
                <span style={{ fontSize: 11, color: "#999", padding: 16, textAlign: "center" }}>
                  Photo / screenshot
                </span>
                <span className="about-media-label">Face Analysis</span>
                <div className="about-media-arrow">↗</div>
              </div>
              <div className="about-media-item" style={{ background: "#e8e8e8" }}>
                <span style={{ fontSize: 11, color: "#999", padding: 16, textAlign: "center" }}>
                  Photo / screenshot
                </span>
                <span
                  className="about-media-label"
                  style={{
                    fontFamily: "var(--font-display)",
                    fontSize: 18,
                    fontWeight: 900,
                    letterSpacing: "-0.02em",
                    bottom: "auto",
                    top: 12,
                    lineHeight: 1.2,
                  }}
                >
                  Modern<br />AI
                </span>
                <div className="about-media-arrow">↗</div>
              </div>
            </div>
          </div>

          <div className="about-right">
            <p className="body-text">
              <strong>MultiMOOD</strong> is a multimodal emotion recognition system developed
              as a diploma project. It is designed for researchers, developers, and practitioners
              who need to analyse emotional states from video by jointly processing text, audio,
              and visual signals.
            </p>
            <p className="body-text">
              The system is trained on <strong>CMU-MOSEI</strong> — a large-scale benchmark
              with over 13,000 samples covering six emotion categories: happiness, sadness,
              anger, surprise, disgust, and fear. The task is formulated as multi-label
              classification, since a single utterance may express more than one emotion.
            </p>
            <p className="body-text">
              The Attention Bottleneck Mechanism allows three encoders — frozen BERT for text,
              Conv1D for audio (COVAREP), and Conv1D for vision (OpenFace) — to exchange
              information through 16 shared learnable tokens, enabling controlled cross-modal
              interaction without full pairwise attention.
            </p>
            <a href="#research" className="btn btn-outline" style={{ marginTop: 8 }}>
              Read more →
            </a>
          </div>
        </section>

        {/* ── EVENTS ── */}
        <section className="events" id="events">
          <div className="events-header">
            <h2 className="section-title">
              Architecture<br />
              <span className="italic">Overview</span>
            </h2>
            <div className="events-desc events-desc-right">
              <p className="events-desc">
                The system processes three modalities through dedicated encoders
                and fuses them via a bottleneck attention mechanism. Each component
                is evaluated independently in the ablation study.
              </p>
              <a href="#research" className="btn btn-outline">View all →</a>
            </div>
          </div>

          <div>
            {EVENTS.map((ev) => (
              <div key={ev.name} className="event-row">
                <div className="event-name">
                  {ev.name}
                  <span className="accent">{ev.accent}</span>
                </div>
                <div className="event-desc">{ev.desc}</div>
                <div className="event-date">{ev.date}</div>
                <div className="event-arrow">↗</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── REVIEWS ── */}
        <section className="reviews" id="reviews">
          <div className="reviews-header">
            <h2 className="reviews-title">
              Key Results &amp;<br />Methodology
            </h2>
            <a href="#" className="btn btn-outline-white">View paper →</a>
          </div>
          <div className="reviews-grid">
            {REVIEWS.map((r) => (
              <div key={r.role} className="review-card">
                <div className="review-img">
                  {/* <Image src="/review1.jpg" alt={r.role} fill style={{ objectFit: "cover" }} /> */}
                  Photo / diagram
                </div>
                <div className="review-body">
                  <div className="review-role">{r.role}</div>
                  <div className="review-text">{r.text}</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── PRICING ── */}
        <section className="pricing" id="pricing">
          <h2 className="pricing-title">Model Comparison</h2>
          <p className="pricing-sub">Baseline vs. proposed architecture vs. ablation variants</p>
          <div className="pricing-grid">
            {PLANS.map((plan) => (
              <div key={plan.tier} className={`pricing-card${plan.featured ? " featured" : ""}`}>
                <div className="pricing-tier">{plan.tier}</div>
                <div className="pricing-price">{plan.price}</div>
                <div className="pricing-period">{plan.period}</div>
                <a
                  href="#"
                  className={`btn pricing-cta ${plan.featured ? "btn-white" : "btn-outline"}`}
                >
                  View details →
                </a>
                <ul className="pricing-features">
                  {plan.features.map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* ── FAQ ── */}
        <FAQ />

        {/* ── FOOTER ── */}
        <footer className="footer">
          <div className="footer-top">
            <div>
              <a href="/" className="footer-logo">MultiMOOD</a>
              <p className="footer-tagline">
                Multimodal Emotion Recognition via Attention Bottleneck Mechanism.
                Diploma project 2025.
              </p>
            </div>
            <div>
              <div className="footer-nav-title">Navigation</div>
              <ul className="footer-nav-list">
                <li><a href="#about">About</a></li>
                <li><a href="#research">Research</a></li>
                <li><a href="#reviews">Results</a></li>
              </ul>
            </div>
            <div>
              <div className="footer-nav-title">More</div>
              <ul className="footer-nav-list">
                <li><a href="#events">Architecture</a></li>
                <li><a href="#">FAQ</a></li>
                <li><a href="https://github.com" target="_blank" rel="noreferrer">GitHub</a></li>
              </ul>
            </div>
          </div>
          <div className="footer-bottom">
            <div>
              <p className="footer-copy">© MultiMOOD 2025</p>
              <p className="footer-copy">All rights reserved</p>
            </div>
            <div className="footer-socials">
              <a href="#" className="footer-social">Instagram</a>
              <a href="#" className="footer-social">GitHub</a>
              <a href="#" className="footer-social">Telegram</a>
            </div>
          </div>
        </footer>

      </main>
    </>
  );
}