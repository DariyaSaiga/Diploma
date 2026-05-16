"use client";

import { useState } from "react";

const FAQS = [
  {
    q: "What is new about your work?",
    a: "The novelty lies in the use of a bottleneck fusion mechanism that allows for efficient fusion of modalities through a limited number of latent tokens instead of full cross-attention, reducing computational complexity and improving generalization.",
  },
  {
    q: "Why exactly Bottleneck?",
    a: "Because classic cross-attention between all tokens is computationally expensive, requiring O(T²). Bottleneck tokens allow for more efficient information compression and transfer between modalities.",
  },
  {
    q: "What data do you use?",
    a: "The CMU-MOSEI datasets are used, containing synchronized text, audio, and visual features with emotion labeling.",
  },
  {
    q: "What emotions does the system detect?",
    a: "The model classifies six emotions: happiness, sadness, anger, surprise, disgust and fear.",
  },
    {
    q: "Why F1 and not accuracy?",
    a: "Because accuracy doesn't work well when classes are imbalanced, while F1 takes into account the balance between precision and recall.",
  },
];

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
          {/* Замените на ваше фото: <Image src="/map.jpg" alt="Маршрут" fill style={{objectFit:'cover'}} /> */}
          <span>Your photo or route map</span>
        </div>
        <p className="faq-media-caption">
          See how easy it is to start working with EmotionAI by following simple instructions in our documentation.
        </p>
      </div>
    </section>
  );
}