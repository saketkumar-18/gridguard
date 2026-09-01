import { ReactElement } from "react";
import { TIER_META, Tier } from "../lib/explain";

/** Horizontal risk gauge: score 0..1 with threshold markers. */
export function Gauge({
  score,
  thresholds,
  tier,
  width = 260,
}: {
  score: number;
  thresholds: { high: number; balanced: number };
  tier: Tier;
  width?: number;
}): ReactElement {
  const h = 10;
  const x = (v: number) => v * width;
  const t = TIER_META[tier];
  return (
    <div className="gauge">
      <svg width={width} height={h + 14} viewBox={`0 0 ${width} ${h + 14}`}>
        <defs>
          <linearGradient id="ggrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#34d399" />
            <stop offset="55%" stopColor="#facc15" />
            <stop offset="100%" stopColor="#f87171" />
          </linearGradient>
        </defs>
        <rect x="0" y="2" width={width} height={h} rx="5" fill="#0b1220" stroke="#1e293b" />
        <rect x="0" y="2" width={width} height={h} rx="5" fill="url(#ggrad)" opacity="0.55" />
        {/* threshold ticks */}
        <line x1={x(thresholds.balanced)} y1="0" x2={x(thresholds.balanced)} y2={h + 4} stroke="#facc15" strokeWidth="1.5" strokeDasharray="2 2" />
        <line x1={x(thresholds.high)} y1="0" x2={x(thresholds.high)} y2={h + 4} stroke="#f87171" strokeWidth="1.5" strokeDasharray="2 2" />
        {/* needle */}
        <circle cx={x(Math.min(Math.max(score, 0), 1))} cy={2 + h / 2} r="5" fill={t.color} stroke="#0b1220" strokeWidth="2" />
      </svg>
      <div className="gauge-legend">
        <span style={{ color: "#facc15" }}>⌄ balanced {(thresholds.balanced * 100).toFixed(0)}%</span>
        <span style={{ color: "#f87171" }}>⌄ high {(thresholds.high * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}
