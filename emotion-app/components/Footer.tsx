const NAV_LEFT = [
  { href: "#about",   label: "About"        },
  { href: "#reviews", label: "Results"      },
  { href: "#events",  label: "Architecture" },
];
const NAV_RIGHT = [
  { href: "#pricing", label: "Comparison" },
  { href: "#faq", label: "FAQ" },
  { href: "https://github.com/DariyaSaiga/Diploma/tree/new_arch", label: "GitHub" },
];
const SOCIALS = ["GitHub"];

export default function Footer() {
  return (
    <footer className="bg-[#111] text-white px-10 pt-12 pb-8 max-md:px-5 max-md:pt-10 max-md:pb-7">
      <div className="grid grid-cols-3 max-md:grid-cols-1 gap-10 max-md:gap-8 pb-10 border-b border-white/10">
        <div>
          <a href="/" className="font-holtwood text-lg text-white no-underline block mb-3">
            MultiMOOD
          </a>
          <p className="text-sm lg:text-base text-white/40 leading-[1.7]">
            Multimodal Emotion Recognition via Attention Bottleneck Mechanism.
            Diploma project 2026.
          </p>
        </div>
        <div>
          <div className="text-xs uppercase tracking-[0.1em] text-white/40 mb-4 font-semibold">
            Navigation
          </div>
          <ul className="flex flex-col gap-3 p-0 list-none">
            {NAV_LEFT.map(({ href, label }) => (
              <li key={label}>
                <a href={href} className="text-sm lg:text-base text-white/70 no-underline hover:text-white transition-colors duration-200">
                  {label}
                </a>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <div className="text-xs uppercase tracking-[0.1em] text-white/40 mb-4 font-semibold">
            More
          </div>
          <ul className="flex flex-col gap-3 p-0 list-none">
            {NAV_RIGHT.map(({ href, label }) => (
              <li key={label}>
                <a
                  href={href}
                  className="text-sm lg:text-base text-white/70 no-underline hover:text-white transition-colors duration-200"
                  {...(href.startsWith("http") ? { target: "_blank", rel: "noreferrer" } : {})}
                >
                  {label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="flex justify-between items-center pt-7 max-md:flex-col max-md:gap-4 max-md:items-start">
        <div>
          <p className="text-sm text-white/30">Development of a multimodal emotion recognition system using attention bottleneck mechanism</p>
          <p className="text-sm text-white/30">Computer Science, IT-2307</p>
          <p className="text-sm text-white/30">Alpieva L., Ablanova D.</p>
        </div>
        <div className="flex gap-3 flex-wrap">
          {SOCIALS.map((label) => (
            <a
            key={label}
            href="https://github.com/DariyaSaiga/Diploma/tree/new_arch"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center px-4 py-2 rounded border border-white/15 text-sm text-white/70 no-underline hover:border-white/50 hover:text-white transition-all duration-200"
          >
            {label}
          </a>
          ))}
        </div>
      </div>
    </footer>
  );
}
