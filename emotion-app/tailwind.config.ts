import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['var(--font-goldman)', 'sans-serif'],
        body: ['var(--font-body)', 'sans-serif'],
        goldman: ['var(--font-goldman)', 'sans-serif'],
        holtwood: ['var(--font-holtwood)', 'serif'],
      },
    },
  },
  plugins: [],
};
export default config;
