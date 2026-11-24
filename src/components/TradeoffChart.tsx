import { policies } from "@/lib/site";

// Speakers cut off (vertical) against how long the agent waits (horizontal).
// Lower-left is better: fewer interruptions, faster replies.
export function TradeoffChart() {
  const W = 560, H = 340, left = 48, right = 16, top = 30, bottom = 44;
  const x = (ms: number) => left + (ms / 1100) * (W - left - right);
  const y = (pct: number) => top + (1 - pct / 90) * (H - top - bottom);
  const timers = policies.filter((p) => !p.ours);
  const ours = policies.filter((p) => p.ours);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img"
      aria-label="Chart: share of speakers cut off against mean wait, for fixed silence timers and for Viraam. The same data is in the table.">
      {[0, 30, 60, 90].map((v) => (
        <g key={v}>
          <line x1={left} x2={W - right} y1={y(v)} y2={y(v)} className="grid" />
          <text x={left - 8} y={y(v) + 4} textAnchor="end">{v}%</text>
        </g>
      ))}
      {[200, 500, 700, 1000].map((v) => (
        <text key={v} x={x(v)} y={H - bottom + 18} textAnchor="middle">{v} ms</text>
      ))}
      <text x={W - right} y={H - 6} textAnchor="end" className="axis">mean wait before replying</text>
      <text x={left} y={12} className="axis">speakers cut off</text>
      <polyline className="timer-line" points={timers.map((p) => `${x(p.wait)},${y(p.cut)}`).join(" ")} />
      {timers.map((p) => (
        <g key={p.name}>
          <circle cx={x(p.wait)} cy={y(p.cut)} r={5} className="timer-dot" />
          <text x={x(p.wait) + 10} y={y(p.cut) - 8}>{p.wait} ms timer</text>
        </g>
      ))}
      {ours.map((p) => (
        <g key={p.name}>
          <circle cx={x(p.wait)} cy={y(p.cut)} r={7} className="ours-dot" />
          <text x={x(p.wait) - 10} y={y(p.cut) + 22} className="ours-label">
            {p.name.replace("Viraam, reply", "Viraam")}
          </text>
        </g>
      ))}
    </svg>
  );
}
