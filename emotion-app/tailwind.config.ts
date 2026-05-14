import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // ── Moon Spell Palette ──────────────────────────────────────────────
      colors: {
        primary:   "#7b5293",
        secondary: "#d9c494",
        accent1:   "#edabf2",
        accent2:   "#bbedef",
        base:      "#fbe3f6",
        bg:        "#0d0a1e",
      },
      fontFamily: {
        display: ["Syne", "sans-serif"],
        body:    ["DM Sans", "sans-serif"],
      },
      backgroundImage: {
        "gradient-moon": "linear-gradient(135deg, #7b5293, #edabf2)",
        "gradient-sand": "linear-gradient(135deg, #d9c494, #fbe3f6)",
        "gradient-ice":  "linear-gradient(135deg, #bbedef, #edabf2)",
      },
      animation: {
        "fade-up":   "fade-up 0.6s cubic-bezier(0.16,1,0.3,1) both",
        "slide-in":  "slide-in 0.35s ease both",
        "orb-drift": "orb-drift 12s ease-in-out infinite alternate",
      },
      keyframes: {
        "fade-up": {
          from: { opacity: "0", transform: "translateY(24px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in": {
          from: { opacity: "0", transform: "translateX(20px)" },
          to:   { opacity: "1", transform: "translateX(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
