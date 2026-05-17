export default function HeroSection() {
  return (
    <section className="hero">
      <div className="hero-media">
        <img
          src="/faces.jpg"
          alt="Emotion recognition"
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
  );
}
