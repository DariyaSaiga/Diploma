import Image from "next/image";
const btnOutline =
  "inline-flex items-center gap-2 px-6 py-3 text-[13px] font-semibold rounded border border-[#111] bg-transparent text-[#111] no-underline cursor-pointer hover:opacity-85 hover:-translate-y-px transition-all duration-200";

export default function AboutSection() {
  return (
    <section
      className="grid grid-cols-1 md:grid-cols-2 px-10 py-20 max-md:px-5 max-md:py-12 border-b border-[#e8e8e8] gap-10 md:gap-0"
      id="about"
    >
      {/* Left column */}
      <div className="flex flex-col gap-8 md:flex-col md:items-start">
        <h2
          className="font-display font-black tracking-[-0.03em] leading-[1.1] md:w-[42%]"
          style={{ fontSize: "clamp(24px, 4vw, 42px)" }}
        >
          About the
          <br />
          <span className="italic font-normal">MultiMOOD</span>
        </h2>

        <div className="md:w-[90%] flex flex-col gap-6">
          <p className="text-md text-[#666] leading-[1.75]">
            <strong className="text-[#111] font-semibold">MultiMOOD</strong> is a
            multimodal emotion recognition system developed as a diploma project.
            It is designed for researchers, developers, and practitioners who need
            to analyse emotional states from video by jointly processing text,
            audio, and visual signals.
          </p>

          <p className="text-md text-[#666] leading-[1.75]">
            The system is trained on{" "}
            <strong className="text-[#111] font-semibold">CMU-MOSEI</strong> — a
            large-scale benchmark with over 13,934 samples covering six emotion
            categories: happiness, sadness, anger, surprise, disgust, and fear.
            The task is formulated as multi-label classification, since a single
            utterance may express more than one emotion.
          </p>

          <p className="text-md text-[#666] leading-[1.75] ">
            The Attention Bottleneck Mechanism allows three encoders — frozen BERT
            for text, Conv1D for audio (COVAREP), and Conv1D for vision (OpenFace)
            — to exchange information through 16 shared learnable tokens, enabling
            controlled cross-modal interaction without full pairwise attention.
          </p>

          <a
            href="#events"
            className="inline-flex h-[50px] w-[180px] items-center justify-center rounded border border-[#111] bg-transparent text-md text-black font-bold uppercase tracking-[0.18em] hover:text-white no-underline transition-all duration-300 hover:border-red-500 hover:bg-red-700 hover:shadow-[0_0_38px_rgba(239,68,68,0.95),0_0_80px_rgba(239,68,68,0.45)]"
          >
            Read more
          </a>
        </div>
      </div>

      {/* Card */}
      <div className="border border-[#e8e8e8] rounded overflow-hidden">
        <div className="inline-flex items-center gap-1.5 text-sm font-semibold bg-[#f5f5f5] px-3 py-1.5 border-b border-[#e8e8e8] w-full">
          <span className="w-1.5 h-1.5 rounded-full bg-[#111] shrink-0" />
          Accuracy &amp; Prediction
        </div>
        <div className="p-5">
          <p className="text-md text-[#666] leading-[1.65]">
            The model predicts six emotion categories — happiness, sadness, anger, surprise, disgust, and fear — by fusing text, audio, and visual signals through an Attention Bottleneck Mechanism, achieving higher accuracy than a standard late fusion baseline
          </p>
        </div>
        <div className="w-full border-t border-[#e8e8e8] flex items-center justify-center overflow-hidden">
          <Image src="/videoPred.png" alt="Accuracy & Prediction" width={800} height={700} />
        </div>
      </div>
      
    </section>
  );
}
