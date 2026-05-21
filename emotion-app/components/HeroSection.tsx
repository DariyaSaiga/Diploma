export default function HeroSection() {
  return (
    <section className="mt-14 px-10 pb-12 max-md:px-4 max-md:pb-8">
    <div className="relative w-full aspect-[16/7] max-md:aspect-[4/5] bg-[#e8e8e8] rounded overflow-hidden">
      <img
        src="/faces.jpg"
        alt="Emotion recognition"
        className="absolute inset-0 h-full w-full object-cover"
      />

      {/* Dark overlay for readability */}
      <div className="absolute inset-0 bg-black/15" />

      {/* Bottom blur up to the middle, smooth fade */}
      {/* <div className="absolute bottom-0 left-0 right-0 h-[58%] backdrop-blur-[12px] bg-black/15 [mask-image:linear-gradient(to_top,black_0%,black_45%,rgba(0,0,0,0.65)_68%,transparent_100%)] [-webkit-mask-image:linear-gradient(to_top,black_0%,black_45%,rgba(0,0,0,0.65)_68%,transparent_100%)]" /> */}

      {/* Smooth lower gradient */}
      <div className="absolute inset-0 bg-[linear-gradient(to_top,rgba(0,0,0,0.62)_0%,rgba(0,0,0,0.32)_45%,transparent_78%)]" />

      {/* Content centered */}
      <div className="relative z-10 flex h-full flex-col items-center justify-center px-8 text-center max-md:px-5">
        <h1
          className="font-goldman font-black text-white tracking-[-0.03em] leading-[1.05] mb-5 drop-shadow-[0_10px_22px_rgba(0,0,0,0.35)]"
          style={{ fontSize: "clamp(44px, 8vw, 118px)" }}
        >
          Understanding
          <br />
          Human Emotions
        </h1>

        <p className="text-white/80 text-lg max-w-[560px] leading-[1.7] mb-8 drop-shadow-[0_4px_12px_rgba(0,0,0,0.25)]">
          A multimodal AI system that recognises emotions from video and audio
          using an Attention Bottleneck Mechanism.
        </p>

        <a
          href="#video"
          className="group relative inline-flex items-center justify-center gap-3 overflow-hidden rounded-full border border-white/45 px-8 py-3.5 text-[13px] font-bold uppercase tracking-[0.18em] text-white no-underline transition-all duration-300 hover:border-red-500 hover:shadow-[0_0_38px_rgba(239,68,68,0.95),0_0_80px_rgba(239,68,68,0.45)]"
        >
          {/* Fill from center */}
          <span className="absolute left-1/2 top-0 h-full w-0 -translate-x-1/2 bg-red-700 transition-all duration-500 ease-out group-hover:w-full" />

          <span className="relative z-10">Recognize</span>

        </a>
      </div>
    </div>
  </section>
);
}
