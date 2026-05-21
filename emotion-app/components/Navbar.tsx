"use client";

import Link from "next/link";
import { useState } from "react";

const NAV_LINKS = [
  { href: "#about",   label: "About"        },
  { href: "#events",  label: "Architecture" },
  { href: "#pricing", label: "Comparison"   },
  { href: "#reviews", label: "Results"      },
];

const linkClass =
  "text-base lg:text-lg text-[#111] no-underline font-medium hover:opacity-50 transition-opacity duration-200";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-[100] bg-white border-b border-[#e8e8e8] flex items-center justify-between px-10 h-14 max-md:px-5">
      <Link href="/" className="font-holtwood text-lg text-[#111] no-underline">
        MultiMOOD
      </Link>

      <div className="hidden md:flex gap-8">
        {NAV_LINKS.map(({ href, label }) => (
          <a key={href} href={href} className={linkClass}>{label}</a>
        ))}
      </div>

      <button
        className="flex md:hidden flex-col justify-center gap-[5px] w-8 h-8 bg-transparent border-none cursor-pointer p-1"
        onClick={() => setMenuOpen((v) => !v)}
        aria-label="Toggle menu"
      >
        <span className="block h-0.5 bg-[#111] rounded-sm transition-all duration-200" />
        <span className="block h-0.5 bg-[#111] rounded-sm transition-all duration-200" />
        <span className="block h-0.5 bg-[#111] rounded-sm transition-all duration-200" />
      </button>

      {menuOpen && (
        <div
          className="fixed top-14 left-0 right-0 bg-white border-b border-[#e8e8e8] flex flex-col z-[99]"
          onClick={() => setMenuOpen(false)}
        >
          {NAV_LINKS.map(({ href, label }) => (
            <a
              key={href}
              href={href}
              className="text-base font-medium text-[#111] no-underline py-4 px-6 border-t border-[#e8e8e8] hover:bg-[#f5f5f5] transition-colors duration-150"
            >
              {label}
            </a>
          ))}
        </div>
      )}
    </nav>
  );
}
