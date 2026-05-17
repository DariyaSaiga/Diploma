"use client";

import { useState } from "react";
import { FAQS } from "@/lib/data";

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  const toggle = (i: number) => setOpenIndex(openIndex === i ? null : i);

  return (
    <section className="faq">
      <div className="faq-left">
        <h2 className="faq-title">Your questions<br />and our answers</h2>
        <div className="faq-list">
          {FAQS.map((item, i) => (
            <div key={i} className="faq-item">
              <button className="faq-question" onClick={() => toggle(i)}>
                <span>{item.q}</span>
                <div className={`faq-toggle${openIndex === i ? " open" : ""}`}>
                  {openIndex === i ? "×" : "+"}
                </div>
              </button>
              <div className={`faq-answer${openIndex === i ? " open" : ""}`}>
                {item.a}
              </div>
            </div>
          ))}
        </div>
        <a href="#pricing" className="btn btn-outline" style={{ marginTop: 24 }}>
          See all →
        </a>
      </div>

      <div className="faq-right">
        <div className="faq-media">
          <span>Your photo or route map</span>
        </div>
        <p className="faq-media-caption">
          See how easy it is to start working with EmotionAI by following simple instructions in our documentation.
        </p>
      </div>
    </section>
  );
}
