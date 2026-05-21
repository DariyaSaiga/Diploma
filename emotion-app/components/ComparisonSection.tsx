import { PLANS } from "@/lib/data";

export default function PricingSection() {
  return (
    <section
      className="px-10 py-20 max-md:px-5 max-md:py-12 border-b border-[#e8e8e8]"
      id="pricing"
    >
      <h2
        className="font-display font-black tracking-[-0.03em] text-center mb-3"
        style={{ fontSize: "clamp(28px, 4vw, 52px)" }}
      >
        Model Comparison
      </h2>
      <p className="text-center text-base lg:text-lg text-[#999] mb-12">
        Baseline vs. proposed architecture vs. ablation variants
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-md:max-w-[480px] max-md:mx-auto">
        {PLANS.map((plan) => (
          <div
            key={plan.tier}
            className={`border rounded px-6 py-7 ${
              plan.featured ? "bg-[#111] text-white border-[#111]" : "border-[#e8e8e8]"
            }`}
          >
            <div
              className={`text-xs uppercase tracking-[0.1em] font-semibold mb-3 ${
                plan.featured ? "text-white/60" : "text-[#696969]"
              }`}
            >
              {plan.tier}
            </div>
            <div className="font-display font-black tracking-[-0.03em] text-3xl md:text-4xl mb-1">
              {plan.price}
            </div>
            <div
              className={`text-sm lg:text-base mb-5 ${
                plan.featured ? "text-white/80" : "text-[#424141]"
              }`}
            >
              {plan.period}
            </div>

            <ul className="flex flex-col gap-2.5 p-0 list-none">
              {plan.features.map((f) => (
                <li
                  key={f}
                  className={`flex items-start gap-2 text-sm lg:text-base leading-[1.5] ${
                    plan.featured ? "text-white/90" : "text-[#383838]"
                  }`}
                >
                  <span
                    className={`text-[6px] mt-1.5 shrink-0 ${
                      plan.featured ? "text-white" : "text-[#111]"
                    }`}
                  >
                    ●
                  </span>
                  {f}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}
