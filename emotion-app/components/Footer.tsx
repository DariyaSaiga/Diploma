const AUTHORS = [
  {
    name: "Dariya Ablanova",
    github: "https://github.com/DariyaSaiga",
    email: "ablanovad@gmail.com",
  },
  {
    name: "Leila Alpieva",
    github: "https://github.com/lyalia123",
    email: "",
  },
];

export default function Footer() {
  return (
    <footer className="bg-[#111] text-white px-10 pt-12 pb-8 max-md:px-5 max-md:pt-10 max-md:pb-7">
      <div className="grid grid-cols-3 max-md:grid-cols-1 gap-10 max-md:gap-8 pb-10 border-b border-white/10">

        {/* Brand */}
        <div className="flex flex-col gap-4">
          <a href="/" className="font-holtwood text-lg text-white no-underline block">
            MultiMOOD
          </a>
          <p className="text-sm lg:text-base text-white/40 leading-[1.7]">
            Multimodal Emotion Recognition based on Attention Bottleneck Mechanism.
            Diploma project 2026.
          </p>
          <a
            href="https://github.com/DariyaSaiga/Diploma/tree/new_arch"
            target="_blank"
            rel="noreferrer"
            className="btn-press inline-flex items-center w-28 gap-2 px-4 py-2 rounded border border-white/15 text-sm text-white/70 no-underline hover:border-red-700 hover:text-white hover:shadow-[0_0_36px_rgba(185,28,28,1),0_0_80px_rgba(185,28,28,0.5)] transition-all duration-200"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            GitHub
          </a>
        </div>

        {/* Authors */}
        {AUTHORS.map((author) => (
          <div key={author.name}>
            <div className="text-sm uppercase tracking-[0.1em] text-white/40 mb-4 font-semibold">
              Author
            </div>
            <p className="text-md lg:text-base text-white font-semibold mb-1">{author.name}</p>
            <p className="text-sm text-white/40 mb-4">Computer Science, IT-2307</p>
            <div className="flex flex-col gap-2">
              <a
                href={author.github}
                target="_blank"
                rel="noreferrer"
                className="btn-press inline-flex items-center gap-2 text-sm text-white/60 no-underline hover:text-white transition-colors duration-200"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
                </svg>
                GitHub
              </a>
              {author.email && (
                <a
                  href={`mailto:${author.email}`}
                  className="btn-press inline-flex items-center gap-2 text-sm text-white/60 no-underline hover:text-white transition-colors duration-200"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="2" y="4" width="20" height="16" rx="2"/>
                    <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
                  </svg>
                  {author.email}
                </a>
              )}
            </div>
          </div>
        ))}

      </div>

    </footer>
  );
}
