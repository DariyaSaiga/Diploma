import { STATS } from "@/lib/data";

export default function StatsSection() {
  return (
    <section className="stats">
      {STATS.map((s) => (
        <div key={s.label} className="stat">
          <div className="stat-value">{s.value}</div>
          <div className="stat-label">{s.label}</div>
        </div>
      ))}
    </section>
  );
}
