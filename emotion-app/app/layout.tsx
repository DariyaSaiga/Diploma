import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EmotionAI — Multimodal Emotion Recognition",
  description: "Multimodal AI emotion recognition system using Attention Bottleneck Mechanism. Diploma project 2025.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Unbounded:wght@400;700;900&family=Manrope:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body>{children}</body>
    </html>
  );
}
