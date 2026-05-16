import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EmotionAI — Мультимодальное распознавание эмоций",
  description: "Мультимодальная AI-система распознавания эмоций на базе механизма внимания (Attention Bottleneck). ResNet-50 + Wav2Vec 2.0.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Unbounded:wght@400;700;900&family=Manrope:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
