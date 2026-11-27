// Everything that names the company or points at a service lives here.
export const site = {
  company: "Viraam",
  tagline: "Voice AI that knows when to speak",
  // The stress-test arena is its own site.
  arena: "https://arena.kognigen.com",
  // Where the public API lives, for the examples on the product page. The API
  // is part of this site (src/app/api). On Vercel the address is worked out
  // from the project's production domain; NEXT_PUBLIC_EOT_API overrides it.
  api: (
    process.env.NEXT_PUBLIC_EOT_API ??
    (process.env.VERCEL_PROJECT_PRODUCTION_URL
      ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}/api`
      : "http://localhost:3000/api")
  ).replace(/\/$/, ""),
};

// Measured results, copied from the detector project's results/ files.
export const policies = [
  { name: "Fixed 300 ms silence", cut: 79.3, wait: 300, ours: false },
  { name: "Fixed 500 ms silence", cut: 55.7, wait: 500, ours: false },
  { name: "Viraam, reply at 50%", cut: 46.8, wait: 218, ours: true },
  { name: "Viraam, reply at 90%", cut: 36.2, wait: 341, ours: true },
  { name: "Fixed 700 ms silence", cut: 24.1, wait: 700, ours: false },
  { name: "Fixed 1,000 ms silence", cut: 7.9, wait: 1000, ours: false },
];

export const awayGame = [
  { language: "Hindi", clips: "1,284", ours: 73.8, smartTurn: 92.8 },
  { language: "Marathi", clips: "774", ours: 68.6, smartTurn: 88.1 },
  { language: "Bengali", clips: "1,000", ours: 68.2, smartTurn: 85.1 },
];
