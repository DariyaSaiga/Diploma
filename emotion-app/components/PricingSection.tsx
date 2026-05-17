import { PLANS } from "@/lib/data";

export default function PricingSection() {
  return (
    <section className="pricing" id="pricing">
      <h2 className="pricing-title">Model Comparison</h2>
      <p className="pricing-sub">Baseline vs. proposed architecture vs. ablation variants</p>
      <div className="pricing-grid">
        {PLANS.map((plan) => (
          <div key={plan.tier} className={`pricing-card${plan.featured ? " featured" : ""}`}>
            <div className="pricing-tier">{plan.tier}</div>
            <div className="pricing-price">{plan.price}</div>
            <div className="pricing-period">{plan.period}</div>
            <a
              href="#"
              className={`btn pricing-cta ${plan.featured ? "btn-white" : "btn-outline"}`}
            >
              View details →
            </a>
            <ul className="pricing-features">
              {plan.features.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}
