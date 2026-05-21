import { STATS } from "@/lib/data";

function getStatBorderClass(i: number) {
  return [
    // border-left: always on i=1 and i=3 (right cell of each row on mobile, all except first on desktop)
    (i === 1 || i === 3) && "border-l",
    // border-top on mobile only for second row (i=2,3)
    (i === 2 || i === 3) && "max-md:border-t",
    // desktop: i=2 also gets border-left (md+)
    i === 2 && "md:border-l",
  ]
    .filter(Boolean)
    .join(" ");
}

export default function StatsSection() {
  return (
    <section className="grid grid-cols-2 md:grid-cols-4 py-9 px-10 max-md:py-6 max-md:px-5 border-t border-b border-[#e8e8e8]">
      {STATS.map((s, i) => (
        <div
          key={s.label}
          className={`px-5 py-0 max-md:p-4 border-[#e8e8e8] ${getStatBorderClass(i)}`}
        >
          <div className="font-display font-black tracking-[-0.03em] text-4xl max-md:text-[28px]">
            {s.value}
          </div>
          <div className="text-lg text-[#999] mt-1 font-medium">{s.label}</div>
        </div>
      ))}
    </section>
  );
}
