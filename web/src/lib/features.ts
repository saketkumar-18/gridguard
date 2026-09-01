/**
 * GridGuard TypeScript feature engine.
 *
 * MIRRORS src/features.py (Python reference) EXACTLY — same imputation, same
 * formulas, same guards, same feature order. Cross-language parity is asserted
 * by tests/data/parity_cases.json goldens produced by scripts/prepare.py.
 * Any change here must be mirrored in Python and regenerate the goldens.
 */

export const FEATURE_SPEC_VERSION = 2;
export const DEFAULT_DOW0 = 3; // JS getDay of 2014-01-01 (Wed)
const MONTH_BUCKET = 30;
const EPS = 1e-9;

export type Maybe = number | null;

export const FEATURE_NAMES: string[] = [
  "mean_cons", "std_cons", "min_cons", "q25_cons", "median_cons",
  "q75_cons", "max_cons", "cv_cons",
  "skew_cons", "kurt_cons",
  "zero_ratio", "long_zero_run",
  "weekday_mean", "weekend_mean", "weekend_ratio",
  "month_spread",
  "half_ratio", "trend_slope",
  "mad_daily", "std_diff", "pct_change_mean",
  "zero_run_norm",
];

export function impute(series: Maybe[]): number[] {
  const xs: (number | null)[] = series.map((v) =>
    v === null ? null : v < 0 ? 0 : v
  );
  const n = xs.length;
  if (n === 0) return [];
  let fv = -1;
  for (let i = 0; i < n; i++) if (xs[i] !== null) { fv = i; break; }
  if (fv === -1) return new Array<number>(n).fill(0);
  for (let k = 0; k < fv; k++) xs[k] = xs[fv];
  let i = fv;
  while (i < n) {
    if (xs[i] === null) {
      let j = i;
      while (j < n && xs[j] === null) j++;
      const left = xs[i - 1] as number;
      if (j === n) {
        for (let k = i; k < j; k++) xs[k] = left;
      } else {
        const right = xs[j] as number;
        const gap = j - i;
        for (let k = i; k < j; k++)
          xs[k] = left + ((right - left) * (k - i + 1)) / (gap + 1);
      }
      i = j;
    } else {
      i++;
    }
  }
  return xs as number[];
}

const mean = (xs: number[]): number =>
  xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;

function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return Math.sqrt(s / (xs.length - 1));
}

function skew(xs: number[]): number {
  const n = xs.length;
  if (n < 3) return 0;
  const m = mean(xs);
  let m2 = 0, m3 = 0;
  for (const x of xs) {
    const d = x - m;
    m2 += d * d;
    m3 += d * d * d;
  }
  m2 /= n; m3 /= n;
  return m2 > 1e-24 ? m3 / Math.pow(m2, 1.5) : 0;
}

function kurt(xs: number[]): number {
  const n = xs.length;
  if (n < 4) return 0;
  const m = mean(xs);
  let m2 = 0, m4 = 0;
  for (const x of xs) {
    const d = x - m;
    m2 += d * d;
    m4 += d * d * d * d;
  }
  m2 /= n; m4 /= n;
  return m2 > 1e-24 ? m4 / (m2 * m2) - 3 : 0;
}

function quantile(xs: number[], p: number): number {
  if (xs.length === 0) return 0;
  const ys = [...xs].sort((a, b) => a - b);
  if (ys.length === 1) return ys[0];
  const h = (ys.length - 1) * p;
  const lo = Math.floor(h);
  const hi = Math.min(lo + 1, ys.length - 1);
  const frac = h - lo;
  return ys[lo] * (1 - frac) + ys[hi] * frac;
}

const cov = (xs: number[]): number => {
  const m = mean(xs);
  return m > 1e-12 ? std(xs) / m : 0;
};

const zeroRatio = (xs: number[]): number =>
  xs.length ? xs.filter((x) => x <= EPS).length / xs.length : 0;

function longestZeroRun(xs: number[]): number {
  let best = 0, run = 0;
  for (const x of xs) {
    if (x <= EPS) {
      run++;
      if (run > best) best = run;
    } else run = 0;
  }
  return best;
}

function slope(xs: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  const mxi = (n - 1) / 2;
  const my = mean(xs);
  let num = 0;
  for (let i = 0; i < n; i++) num += (i - mxi) * (xs[i] - my);
  let den = 0;
  for (let i = 0; i < n; i++) den += (i - mxi) * (i - mxi);
  return den > 1e-12 ? num / den : 0;
}

function meanAbsDiff(a: number[], b: number[]): number {
  if (!a.length) return 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(b[i] - a[i]);
  return s / a.length;
}

function meanAbsPctChange(a: number[], b: number[]): number {
  if (!a.length) return 0;
  const vals: number[] = [];
  for (let i = 0; i < a.length; i++)
    if (Math.abs(a[i]) > 1e-9) vals.push(Math.abs(b[i] - a[i]) / Math.abs(a[i]));
  return vals.length ? mean(vals) : 0;
}

export function featureVector(series: Maybe[], dow0: number = DEFAULT_DOW0): number[] {
  const x = impute(series);
  const n = x.length;
  if (n === 0) return FEATURE_NAMES.map(() => 0);

  const mu = mean(x);
  const sd = std(x);

  const wd: number[] = [], we: number[] = [];
  for (let i = 0; i < n; i++) {
    const dow = (dow0 + i) % 7;
    (dow === 0 || dow === 6 ? we : wd).push(x[i]);
  }
  const wdMu = wd.length ? mean(wd) : 0;
  const weMu = we.length ? mean(we) : 0;
  const weRatio = wdMu > 1e-12 ? weMu / wdMu : weMu <= 1e-12 ? 1 : 2;

  const nb = Math.ceil(n / MONTH_BUCKET);
  const bucketMeans: number[] = [];
  for (let b = 0; b < nb; b++) {
    const seg = x.slice(b * MONTH_BUCKET, (b + 1) * MONTH_BUCKET);
    if (seg.length) bucketMeans.push(mean(seg));
  }
  const monthSpread =
    bucketMeans.length >= 2 && mu > 1e-12
      ? (Math.max(...bucketMeans) - Math.min(...bucketMeans)) / mu
      : 0;

  const h = Math.floor(n / 2);
  const m1 = mean(x.slice(0, h));
  const m2 = h > 0 ? mean(x.slice(h)) : 0;
  const halfRatio = m1 > 1e-12 ? m2 / m1 : m2 <= 1e-12 ? 1 : 2;

  let prev: number[], nxt: number[], diffs: number[];
  if (n > 1) {
    prev = x.slice(0, n - 1);
    nxt = x.slice(1);
    diffs = nxt.map((v, i) => v - prev[i]);
  } else {
    prev = nxt = diffs = [0];
  }

  const lzr = longestZeroRun(x);

  return [
    mu, sd, Math.min(...x), quantile(x, 0.25), quantile(x, 0.5),
    quantile(x, 0.75), Math.max(...x),
    cov(x),
    skew(x), kurt(x),
    zeroRatio(x), lzr,
    wdMu, weMu, weRatio,
    monthSpread,
    halfRatio, slope(x),
    meanAbsDiff(prev, nxt), std(diffs), meanAbsPctChange(prev, nxt),
    lzr / n,
  ];
}
