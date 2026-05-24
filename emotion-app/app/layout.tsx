import type { Metadata } from "next";
import {
  Goldman,
  Holtwood_One_SC,
  Unbounded,
  Manrope,
} from "next/font/google";
import "./globals.css";
import Providers from "@/components/Providers";

const goldman = Goldman({
  weight: ["400", "700"],
  subsets: ["latin"],
  variable: "--font-goldman",
  display: "swap",
});

const holtwood = Holtwood_One_SC({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-holtwood",
  display: "swap",
});

const unbounded = Unbounded({
  weight: ["400", "700", "900"],
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const manrope = Manrope({
  weight: ["300", "400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "EmotionAI — Multimodal Emotion Recognition",
  description:
    "Multimodal AI emotion recognition system using Attention Bottleneck Mechanism. Diploma project 2025.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${goldman.variable} ${holtwood.variable} ${unbounded.variable} ${manrope.variable}`}
    >
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
