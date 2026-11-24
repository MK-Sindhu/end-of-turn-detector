// The page's atmosphere: a warm-to-cool gradient, fine wavy contour lines
// mirrored around the centre, and film grain over the top.

function waves() {
  const paths: string[] = [];
  for (let i = 0; i < 22; i++) {
    const offset = 30 + i * 34;
    let d = "";
    for (let x = 0; x <= 600; x += 12) {
      const y = offset + x * 0.62 + Math.sin((x + i * 18) / 46) * 13;
      d += `${x === 0 ? "M" : "L"}${x},${y.toFixed(1)} `;
    }
    paths.push(d);
  }
  return paths;
}

export function Backdrop({ variant = "ember" }: { variant?: "ember" | "dusk" }) {
  const lines = waves();
  const quadrants = ["", "scale(-1,1)", "scale(1,-1)", "scale(-1,-1)"];
  return (
    <div className={`backdrop backdrop-${variant}`} aria-hidden="true">
      <svg viewBox="-600 -420 1200 840" preserveAspectRatio="xMidYMid slice">
        {quadrants.map((transform) => (
          <g key={transform} transform={transform}>
            {lines.map((d, i) => (
              <path key={i} d={d} transform="translate(0,-420)" />
            ))}
          </g>
        ))}
        <path d="M-600,0 H600 M0,-420 V420" />
      </svg>
      <div className="grain" />
    </div>
  );
}
