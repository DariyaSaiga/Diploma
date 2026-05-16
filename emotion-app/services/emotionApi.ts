// Emotion API service — connects to FastAPI backend or returns mock data

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const EMOTIONS = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral', 'Contempt'];

function mockPrediction() {
  const primary = EMOTIONS[Math.floor(Math.random() * EMOTIONS.length)];
  const scores = EMOTIONS.map(label => ({
    label,
    score: label === primary
      ? 0.45 + Math.random() * 0.45
      : Math.random() * 0.25,
  }));
  // Normalize
  const total = scores.reduce((s, e) => s + e.score, 0);
  return {
    emotion: primary,
    confidence: scores.find(e => e.label === primary)!.score / total,
    allEmotions: scores.map(e => ({ ...e, score: e.score / total }))
      .sort((a, b) => b.score - a.score),
  };
}

export async function analyzeImage(file: File) {
  await new Promise(r => setTimeout(r, 1400 + Math.random() * 800));
  // Real: const form = new FormData(); form.append('file', file); fetch(`${BASE_URL}/analyze/image`, { method: 'POST', body: form })
  return mockPrediction();
}

export async function analyzeVideo(file: File) {
  await new Promise(r => setTimeout(r, 2200 + Math.random() * 1000));
  return mockPrediction();
}

export async function analyzeAudio(blob: Blob) {
  await new Promise(r => setTimeout(r, 1600 + Math.random() * 800));
  return mockPrediction();
}

export function createWebSocket(onMessage: (data: { emotion: string; confidence: number; allEmotions: { label: string; score: number }[] }) => void) {
  // Real WS: const ws = new WebSocket(`ws://localhost:8000/ws/camera`);
  // Mock: emit random predictions every 1.5s
  let interval: ReturnType<typeof setInterval>;
  const mock = {
    start() {
      interval = setInterval(() => {
        onMessage(mockPrediction());
      }, 1500);
    },
    stop() { clearInterval(interval); },
  };
  return mock;
}
