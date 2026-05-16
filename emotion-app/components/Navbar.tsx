import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="nav">
      <Link href="/" className="nav-logo">EmotionAI</Link>
      <div className="nav-links">
        <a href="#about">О системе</a>
        <a href="#events">Применение</a>
        <a href="#pricing">Тарифы</a>
        <a href="#reviews">Отзывы</a>
      </div>
    </nav>
  );
}
