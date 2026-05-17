export default function Footer() {
  return (
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
            <li><a href="#reviews">Results</a></li>
            <li><a href="#events">Architecture</a></li>
          </ul>
        </div>
        <div>
          <div className="footer-nav-title">More</div>
          <ul className="footer-nav-list">
            <li><a href="#pricing">Comparison</a></li>
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
  );
}
