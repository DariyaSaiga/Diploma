"use client";

import { useState } from "react";
import { FAQS } from "@/lib/data";
import Image from "next/image";

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(0);
  const toggle = (i: number) => setOpenIndex(openIndex === i ? null : i);

  return (
    <section className="grid grid-cols-1 md:grid-cols-2 px-10 py-20 max-md:px-5 max-md:py-12 border-b border-[#e8e8e8] gap-10">
      {/* Left: Q&A */}
      <div>
        <h2
          className="font-display font-black tracking-[-0.03em] leading-[1.1] mb-8"
          style={{ fontSize: "clamp(28px, 3.5vw, 48px)" }}
        >
          Your questions<br />and our answers
        </h2>

        <div className="flex flex-col">
          {FAQS.map((item, i) => (
            <div key={i} className="border-t border-[#e8e8e8]">
              <button
                className="flex items-center justify-between w-full py-5 text-lg lg:text-xl font-semibold text-[#111] cursor-pointer bg-transparent border-none text-left gap-4 hover:text-[#707070] transition-colors duration-200 font-body"
                onClick={() => toggle(i)}
              >
                <span>{item.q}</span>
                <div
                  className={`w-7 h-7 rounded-full border flex items-center justify-center text-base shrink-0 cursor-pointer leading-none transition-all duration-200 ${
                    openIndex === i
                      ? "bg-[#111] text-white border-[#111]"
                      : "border-[#ccc] text-[#111]"
                  }`}
                >
                  {openIndex === i ? "×" : "+"}
                </div>
              </button>
              <div className={`faq-answer${openIndex === i ? " open" : ""}`}>
                {item.a}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right: image + caption */}
      <div className="md:pl-[60px]">
        <div className="relative aspect-[4/3] bg-[#e8e8e8] rounded mb-5 flex items-center justify-center text-[#999] text-sm overflow-hidden">
          <Image
            src="/bazaart.png"
            alt="Us"
            width={400}
            height={300}
            className="object-cover w-full h-full"
          />
      
          <div
            className="absolute top-6 left-6 z-10 -rotate-[10deg]"
            style={{
              textAlign: "center",
              fontFamily: "'Press Start 2P', monospace",
            }}
          >

            

          </div>
        </div>
            
        <p className="text-base lg:text-lg text-[#525252] leading-[1.6]">
          See how easy it is to start working with{" "}
          <span className="font-holtwood text-red-700">MultiMOOD</span>{" "}
          and recognize human emotions through multimodal AI.
        </p>
      </div>
    </section>
  );
}
