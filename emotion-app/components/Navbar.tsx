"use client";

import Link from "next/link";
import { useState } from "react";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="nav">
      <Link href="/" className="nav-logo">EmotionAI</Link>

      {/* Desktop links */}
      <div className="nav-links">
        <a href="#about">About</a>
        <a href="#events">Architecture</a>
        <a href="#pricing">Comparison</a>
        <a href="#reviews">Results</a>
      </div>

      {/* Mobile hamburger */}
      <button
        className="nav-burger"
        onClick={() => setMenuOpen((v) => !v)}
        aria-label="Toggle menu"
      >
        <span /><span /><span />
      </button>

      {/* Mobile dropdown */}
      {menuOpen && (
        <div className="nav-mobile" onClick={() => setMenuOpen(false)}>
          <a href="#about">About</a>
          <a href="#events">Architecture</a>
          <a href="#pricing">Comparison</a>
          <a href="#reviews">Results</a>
        </div>
      )}
    </nav>
  );
}
