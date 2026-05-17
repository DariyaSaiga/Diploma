export default function AboutSection() {
  return (
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
        <a href="#events" className="btn btn-outline" style={{ marginTop: 8 }}>
          Read more →
        </a>
      </div>
    </section>
  );
}
