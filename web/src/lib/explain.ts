/** Shared types + meta loading + tier/explanation logic. */

export interface ModelMeta {
  model: string;
  n_features: number;
  feature_names: string[];
  window_days: number;
  thresholds: { high: number; balanced: number };
  metrics: {
    window: { roc_auc: number; pr_auc: number };
    consumer_mean: Record<string, number | number[][] | string>;
    consumer_max: Record<string, number | number[][] | string>;
  };
  cv_table: { model: string; cv_roc_auc: number; cv_pr_auc: number }[];
  importances: Record<string, number>;
  honest_median: Record<string, number>;
  honest_q25: Record<string, number>;
  honest_q75: Record<string, number>;
  precision_target: number;
}

export interface Consumer {
  id: string;
  flag: number; // SGCC ground truth (demo only)
  score: number; // model score
  series: (number | null)[];
}

export type Tier = "critical" | "high" | "watch" | "ok";

export function tierOf(score: number, t: { high: number; balanced: number }): Tier {
  if (score >= t.high) return "critical";
  if (score >= t.balanced) return "high";
  if (score >= t.balanced * 0.5) return "watch";
  return "ok";
}

export const TIER_META: Record<Tier, { label: string; color: string; note: string }> = {
  critical: {
    label: "CRITICAL",
    color: "#f87171",
    note: "High-confidence anomaly — prioritize field inspection",
  },
  high: {
    label: "HIGH",
    color: "#fb923c",
    note: "Anomalous pattern — secondary inspection queue",
  },
  watch: {
    label: "WATCH",
    color: "#facc15",
    note: "Mild deviation — monitor over next windows",
  },
  ok: {
    label: "OK",
    color: "#34d399",
    note: "Within honest-consumer behavior bands",
  },
};

/** Plain-English descriptions for feature deviations (viva-friendly). */
const FEATURE_EXPLAIN: Record<string, { low: string; high: string }> = {
  mean_cons: { low: "overall usage far below similar honest meters", high: "unusually high overall usage" },
  std_cons: { low: "consumption unusually flat/regular", high: "highly erratic day-to-day usage" },
  min_cons: { low: "near-zero floors (possible bypass periods)", high: "high floor usage" },
  cv_cons: { low: "very low relative volatility", high: "very high relative volatility" },
  skew_cons: { low: "skewed toward low readings", high: "skewed toward heavy spikes" },
  kurt_cons: { low: "few extreme days", high: "extreme spike days present" },
  zero_ratio: { low: "few zero days", high: "many zero-consumption days (meter dead / bypass)" },
  long_zero_run: { low: "no long dead spells", high: "long dead spell (meter off while premise active)" },
  zero_run_norm: { low: "no long dead spells", high: "long dead spell relative to window" },
  weekday_mean: { low: "low weekday usage", high: "high weekday usage" },
  weekend_mean: { low: "low weekend usage", high: "high weekend usage" },
  weekend_ratio: { low: "weekend usage far below weekdays", high: "weekend usage far above weekdays" },
  month_spread: { low: "stable across months", high: "usage shifts sharply between months" },
  half_ratio: { low: "usage dropped in recent half (possible tamper onset)", high: "usage rose in recent half" },
  trend_slope: { low: "declining trend across window", high: "rising trend across window" },
  mad_daily: { low: "smooth daily profile", high: "jumpy daily profile" },
  std_diff: { low: "consistent day-to-day change", high: "wild day-to-day swings" },
  pct_change_mean: { low: "stable relative change", high: "large relative day-to-day jumps" },
  median_cons: { low: "typical day usage very low", high: "typical day usage very high" },
  q75_cons: { low: "even peak days are low", high: "high peak-day usage" },
  max_cons: { low: "no high spikes", high: "extreme single-day spikes" },
  q25_cons: { low: "many near-zero days", high: "floor of usage is high" },
};

export interface FeatureRow {
  name: string;
  value: number;
  q25: number;
  median: number;
  q75: number;
  deviation: number; // signed, in robust units of (q75-q25)
  outsideBand: boolean;
  importance: number;
}

export function featureRows(
  features: number[],
  meta: ModelMeta,
  names: string[]
): FeatureRow[] {
  return names.map((n, i) => {
    const v = features[i];
    const q25 = meta.honest_q25[n] ?? 0;
    const q75 = meta.honest_q75[n] ?? 0;
    const med = meta.honest_median[n] ?? 0;
    const iqr = Math.max(q75 - q25, 1e-9);
    const deviation = (v - med) / iqr;
    return {
      name: n,
      value: v,
      q25,
      median: med,
      q75,
      deviation,
      outsideBand: v < q25 || v > q75,
      importance: meta.importances[n] ?? 0,
    };
  });
}

export function topExplanations(rows: FeatureRow[], k = 4): { text: string; weight: number }[] {
  const cand = rows
    .filter((r) => r.outsideBand && FEATURE_EXPLAIN[r.name])
    .map((r) => ({
      text:
        r.deviation > 0
          ? FEATURE_EXPLAIN[r.name].high
          : FEATURE_EXPLAIN[r.name].low,
      weight: Math.abs(r.deviation) * (0.2 + r.importance),
      name: r.name,
      value: r.value,
    }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, k);
  return cand.map((c) => ({ text: c.text, weight: c.weight }));
}

export const fmt = (v: number, d = 3): string => {
  if (!isFinite(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  return v.toFixed(d);
};
