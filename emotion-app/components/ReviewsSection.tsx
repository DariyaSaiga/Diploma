import { REVIEWS } from "@/lib/data";

export default function ReviewsSection() {
  return (
    <section className="reviews" id="reviews">
      <div className="reviews-header">
        <h2 className="reviews-title">
          Key Results &amp;<br />Methodology
        </h2>
        <a href="#" className="btn btn-outline-white">View paper →</a>
      </div>
      <div className="reviews-grid">
        {REVIEWS.map((r) => (
          <div key={r.role} className="review-card">
            <div className="review-img">
              Photo / diagram
            </div>
            <div className="review-body">
              <div className="review-role">{r.role}</div>
              <div className="review-text">{r.text}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
