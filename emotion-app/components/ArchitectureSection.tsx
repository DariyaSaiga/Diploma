import { EVENTS } from "@/lib/data";

export default function EventsSection() {
  return (
    <section
      className="px-10 py-20 max-md:px-5 max-md:py-12 border-b border-[#e8e8e8]"
      id="events"
    >
      <div className="flex justify-between items-start mb-10 max-md:flex-col max-md:gap-5">
        <h2
          className="font-display font-black tracking-[-0.03em] leading-[1.1]"
          style={{ fontSize: "clamp(28px, 4vw, 48px)" }}
        >
          Architecture<br />
          <span className="italic font-normal">Overview</span>
        </h2>
        <div className="flex flex-col items-end gap-4 text-right max-md:text-left max-md:items-start">
          <p className="text-base lg:text-lg text-[#666] max-w-[420px] leading-[1.7]">
            The system processes three modalities through dedicated encoders and
            fuses them via a bottleneck attention mechanism. Each component is
            evaluated independently in the ablation study.
          </p>
          <a
            href="#reviews"
            className="inline-flex h-[50px] w-[140px] items-center justify-center rounded border border-[#111] bg-transparent text-md text-black font-bold uppercase tracking-[0.18em] hover:text-white no-underline transition-all duration-300 hover:border-red-500 hover:bg-red-700 hover:shadow-[0_0_38px_rgba(239,68,68,0.95),0_0_80px_rgba(239,68,68,0.45)]"
          >
            View all
          </a>
        </div>
      </div>

      <div>
        {EVENTS.map((ev) => (
          <div
            key={ev.name}
            className="group grid grid-cols-[2fr_3fr_1fr_40px] max-md:grid-cols-[1fr_40px] items-center gap-6 max-md:gap-3 py-5 border-t border-[#e8e8e8] last:border-b cursor-pointer hover:bg-[#f5f5f5] hover:-mx-10 hover:px-10 max-md:hover:-mx-5 max-md:hover:px-5 transition-all duration-150"
          >
            <div className="text-base lg:text-lg font-bold tracking-[-0.01em] leading-[1.3]">
              {ev.name}
              <span className="italic font-normal text-[#666] block text-sm lg:text-base">{ev.accent}</span>
            </div>
            <div className="text-sm lg:text-base text-[#666] leading-[1.6] max-md:hidden">{ev.desc}</div>
            <div className="text-sm lg:text-base font-semibold text-[#666] max-md:hidden">{ev.date}</div>
            <div className="w-7 h-7 rounded-full border border-[#ccc] flex items-center justify-center text-sm cursor-pointer transition-all duration-200 group-hover:bg-[#111] group-hover:text-white group-hover:border-[#111]">
              ↗
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
