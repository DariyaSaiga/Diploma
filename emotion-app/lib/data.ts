// ── Static data constants — edit here to update content across the site ──────

export const STATS = [
  { value: "6", label: "Emotion classes" },
  { value: "3", label: "Modalities (text, audio, video)" },
  { value: "87%", label: "Macro-F1 on CMU-MOSEI" },
  { value: "13,934", label: "Training samples" },
];

export const EVENTS = [
  {
    name: "Text Modality",
    accent: "BERT Encoder",
    desc: "Frozen bert-base-uncased extracts the mean of the last 4 hidden layers into a [B, 50, 768] tensor, providing rich contextual language representations without fine-tuning.",
    date: "Implemented",
  },
  {
    name: "Audio Modality",
    accent: "COVAREP Features",
    desc: "74-dimensional acoustic low-level descriptors per frame, fixed to 60 frames per sample. Processed via Conv1D projection and positional encoding into a [B, 60, 128] tensor.",
    date: "Implemented",
  },
  {
    name: "Visual Modality",
    accent: "OpenFace AU",
    desc: "35-dimensional facial action unit features per frame extracted by OpenFace, fixed to 60 frames. Projected with Conv1D and positional encoding into a [B, 60, 128] tensor.",
    date: "Implemented",
  },
  {
    name: "Bottleneck Fusion",
    accent: "Cross-modal Attention",
    desc: "16 learnable bottleneck tokens act as a compact communication channel across all three modalities over 2 attention layers, enabling controlled cross-modal interaction without full pairwise attention.",
    date: "Core module",
  },
];

export const REVIEWS = [
  {
    role: "Dataset — CMU-MOSEI",
    text: "Large-scale multimodal benchmark with 13,934 training / 1,569 validation / 3,957 test samples across 6 emotion categories: happiness, sadness, anger, surprise, disgust, and fear.",
  },
  {
    role: "Training — AdamW + Cosine Annealing",
    text: "Batch size 8, learning rate 1e-4, gradient clipping at 1.0, early stopping on validation Macro-F1 with patience 10. Weighted BCEWithLogitsLoss handles severe class imbalance (fear: 9.6%, happiness: 62.7%).",
  },
  {
    role: "Evaluation — Macro-F1 & Avg. WA",
    text: "Primary metric is macro-averaged F1 across all 6 classes. Per-class threshold tuning on validation set further improves minority-class detection, especially for fear and surprise.",
  },
];

export const PLANS = [
  {
    tier: "Late Fusion Baseline",
    price: "Baseline",
    period: "Independent unimodal classifiers combined at decision level",
    featured: false,
    features: [
      "Separate text, audio, vision heads",
      "Mean-pooled unimodal representations",
      "Simple concatenation before classifier",
      "No cross-modal information exchange",
      "Lower Macro-F1 on minority classes",
    ],
  },
  {
    tier: "Bottleneck Fusion Model",
    price: "Proposed",
    period: "Cross-modal attention via shared learnable bottleneck tokens",
    featured: true,
    features: [
      "Frozen BERT + Conv1D audio & visual encoders",
      "16 learnable bottleneck tokens (hidden dim 128)",
      "2-layer cross-modal attention fusion",
      "Auxiliary unimodal classification heads",
      "Modality dropout (p=0.05) for robustness",
      "Per-class threshold tuning at inference",
    ],
  },
  {
    tier: "Ablation Variants",
    price: "Analysis",
    period: "Systematic component-wise evaluation",
    featured: false,
    features: [
      "Single BERT layer vs. mean of last 4 layers",
      "Modality dropout at p=0, 0.05, 0.10, 0.15",
      "With vs. without auxiliary losses",
      "Default threshold 0.5 vs. tuned thresholds",
      "Bottleneck vs. late fusion comparison",
    ],
  },
];

export const FAQS = [
  {
    q: "What is new about your work?",
    a: "The novelty lies in the use of a bottleneck fusion mechanism that allows for efficient fusion of modalities through a limited number of latent tokens instead of full cross-attention, reducing computational complexity and improving generalization.",
  },
  {
    q: "Why exactly Bottleneck?",
    a: "Because classic cross-attention between all tokens is computationally expensive, requiring O(T²). Bottleneck tokens allow for more efficient information compression and transfer between modalities.",
  },
  {
    q: "What data do you use?",
    a: "The CMU-MOSEI datasets are used, containing synchronized text, audio, and visual features with emotion labeling.",
  },
  {
    q: "What emotions does the system detect?",
    a: "The model classifies six emotions: happiness, sadness, anger, surprise, disgust and fear.",
  },
  {
    q: "Why F1 and not accuracy?",
    a: "Because accuracy doesn't work well when classes are imbalanced, while F1 takes into account the balance between precision and recall.",
  },
];
