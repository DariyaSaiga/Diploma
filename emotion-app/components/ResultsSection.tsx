import { REVIEWS } from "@/lib/data";

export default function ReviewsSection() {
  return (
    <section
      className="bg-[#111] text-white px-10 py-20 max-md:px-5 max-md:py-12"
      id="reviews"
    >
      <div className="flex justify-between items-center mb-10 max-md:flex-col max-md:items-start max-md:gap-5">
        <h2
          className="font-display font-black tracking-[-0.03em] leading-[1.15]"
          style={{ fontSize: "clamp(28px, 3vw, 42px)" }}
        >
          Key Results &amp;<br />Methodology
        </h2>
        <a
          href="#"
          className="inline-flex items-center gap-2 px-6 py-3 text-base font-semibold rounded cursor-pointer no-underline whitespace-nowrap bg-transparent text-white border border-white/50 hover:opacity-85 hover:-translate-y-px transition-all duration-200"
        >
          View paper →
        </a>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {REVIEWS.map((r) => (
          <div
            key={r.role}
            className="bg-white/[0.06] border border-white/10 rounded overflow-hidden"
          >
            <div className="w-full aspect-[4/3] bg-white/[0.08] flex items-center justify-center text-white/20 text-sm overflow-hidden">
              Photo / diagram
            </div>
            <div className="p-5">
              <div className="text-xs text-white/40 uppercase tracking-[0.08em] mb-2">
                {r.role}
              </div>
              <div className="text-base lg:text-lg text-white/85 leading-[1.65] font-medium">
                {r.text}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
