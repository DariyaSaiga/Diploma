// components/Providers.tsx
'use client';
import { EmotionProvider } from '@/lib/EmotionContex';
export default function Providers({ children }: { children: React.ReactNode }) {
  return <EmotionProvider>{children}</EmotionProvider>;
}