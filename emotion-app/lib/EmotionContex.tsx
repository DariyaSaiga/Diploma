'use client';

import React, { createContext, useContext, useState, ReactNode } from 'react';

export interface EmotionResult {
  emotion: string;
  confidence: number;
  allEmotions: { label: string; score: number }[];
  source: 'camera' | 'upload' | 'audio';
  timestamp: Date;
}

interface EmotionContextType {
  results: EmotionResult[];
  currentResult: EmotionResult | null;
  addResult: (result: EmotionResult) => void;
  clearResults: () => void;
  isProcessing: boolean;
  setIsProcessing: (v: boolean) => void;
}

const EmotionContext = createContext<EmotionContextType | null>(null);

export function EmotionProvider({ children }: { children: ReactNode }) {
  const [results, setResults] = useState<EmotionResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const addResult = (result: EmotionResult) => {
    setResults(prev => [result, ...prev.slice(0, 9)]);
  };

  const clearResults = () => setResults([]);
  const currentResult = results[0] ?? null;

  return (
    <EmotionContext.Provider value={{ results, currentResult, addResult, clearResults, isProcessing, setIsProcessing }}>
      {children}
    </EmotionContext.Provider>
  );
}

export function useEmotion() {
  const ctx = useContext(EmotionContext);
  if (!ctx) throw new Error('useEmotion must be used within EmotionProvider');
  return ctx;
}
