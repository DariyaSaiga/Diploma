import { EVENTS } from "@/lib/data";

export default function EventsSection() {
  return (
    <section className="events" id="events">
      <div className="events-header">
        <h2 className="section-title">
          Architecture<br />
          <span className="italic">Overview</span>
        </h2>
        <div className="events-desc events-desc-right">
          <p className="events-desc">
            The system processes three modalities through dedicated encoders
            and fuses them via a bottleneck attention mechanism. Each component
            is evaluated independently in the ablation study.
          </p>
          <a href="#reviews" className="btn btn-outline">View all →</a>
        </div>
      </div>

      <div>
        {EVENTS.map((ev) => (
          <div key={ev.name} className="event-row">
            <div className="event-name">
              {ev.name}
              <span className="accent">{ev.accent}</span>
            </div>
            <div className="event-desc">{ev.desc}</div>
            <div className="event-date">{ev.date}</div>
            <div className="event-arrow">↗</div>
          </div>
        ))}
      </div>
    </section>
  );
}
