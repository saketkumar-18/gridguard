import { ReactElement } from "react";

/** Compact SVG sparkline of a 90-day consumption series. */
export function Sparkline({
  series,
  width = 160,
  height = 36,
  color = "#38bdf8",
  highlight = false,
}: {
  series: (number | null)[];
  width?: number;
  height?: number;
  color?: string;
  highlight?: boolean;
}): ReactElement {
  const pts = series.map((v, i) => ({ v, i })).filter((p) => p.v !== null) as {
    v: number;
    i: number;
  }[];
  const max = Math.max(...pts.map((p) => p.v), 1e-9);
  const n = series.length || 1;
  const x = (i: number) => (i / (n - 1)) * (width - 2) + 1;
  const y = (v: number) => height - 2 - (v / max) * (height - 6);
  const d = pts.map((p, k) => `${k === 0 ? "M" : "L"}${x(p.i).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
  const areaId = `sa${Math.random().toString(36).slice(2, 8)}`;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden>
      <defs>
        <linearGradient id={areaId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      {pts.length > 1 && (
        <>
          <path d={`${d} L${x(pts[pts.length - 1].i).toFixed(1)},${height} L${x(pts[0].i).toFixed(1)},${height} Z`} fill={`url(#${areaId})`} />
          <path d={d} fill="none" stroke={highlight ? "#f87171" : color} strokeWidth="1.4" strokeLinejoin="round" />
        </>
      )}
    </svg>
  );
}
