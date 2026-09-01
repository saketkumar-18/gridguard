"""Declarative feature engineering for smart-meter consumption series.

The SAME definition drives (a) Python training/prediction and (b) the
TypeScript web app: `export_spec()` -> JSON -> web/src/lib/features.ts.
One source of truth guarantees cross-language parity; tests assert it
bit-for-bit on random series.

Each consumer is a daily consumption series x[0..n-1] (kWh) with possible
missing days (None/NaN). Imputation happens BEFORE any statistic.

22 features, families:
  A. Consumption level   mean, std, min, q25, median, q75, max, CV
  B. Shape / tails       skew, kurtosis (population moments)
  C. Zero behaviour      zero_ratio, longest zero run (+ normalized)
  D. Temporal            weekday/weekend means, weekend ratio, month spread
  E. Drift               first-half vs second-half ratio, OLS trend slope
  F. Volatility          mean abs daily diff, std of diffs, mean abs pct change
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

FEATURE_SPEC_VERSION = 2

# Weekday index (JS Date.getDay convention, 0=Sunday) of day 0 of the series.
# SGCC series starts 2014-01-01 = Wednesday = 3.
DEFAULT_DOW0 = 3

MONTH_BUCKET = 30  # days per "month" bucket for month_spread

# ---------------------------------------------------------------------------
# Spec (single source of truth consumed by the TypeScript implementation)
# ---------------------------------------------------------------------------

def build_imputer_spec() -> Dict[str, Any]:
    return {
        "kind": "linear_interp_clip0",
        "interior": "linear",
        "edge": "nearest",
        "all_missing": 0.0,
    }


def export_spec(dow0: int = DEFAULT_DOW0) -> Dict[str, Any]:
    return {
        "version": FEATURE_SPEC_VERSION,
        "imputer": build_imputer_spec(),
        "dow0": dow0,
        "month_bucket": MONTH_BUCKET,
        "features": feature_names(),
    }


def spec_json(dow0: int = DEFAULT_DOW0) -> str:
    return json.dumps(export_spec(dow0=dow0), separators=(",", ":"))


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute(series: List[Optional[float]]) -> List[float]:
    """clip negatives to 0 -> linear interpolation over interior gaps ->
    nearest-neighbour fill for edge gaps -> 0 for an all-missing series."""
    xs: List[Optional[float]] = [
        (0.0 if v < 0 else float(v)) if v is not None else None for v in series
    ]
    n = len(xs)
    if n == 0:
        return []
    fv = next((i for i, v in enumerate(xs) if v is not None), None)
    if fv is None:
        return [0.0] * n
    for k in range(fv):                       # leading gap <- first valid
        xs[k] = xs[fv]
    i = fv
    while i < n:
        if xs[i] is None:
            j = i
            while j < n and xs[j] is None:
                j += 1
            left = xs[i - 1]
            if j == n:                        # trailing gap -> last valid
                for k in range(i, j):
                    xs[k] = left
            else:                             # interior gap -> linear interp
                right = xs[j]
                gap = j - i
                for k in range(i, j):
                    xs[k] = left + (right - left) * (k - i + 1) / (gap + 1)
            i = j
        else:
            i += 1
    return [float(v) for v in xs]


# ---------------------------------------------------------------------------
# Statistical primitives (pure; TS mirrors exactly)
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _skew(xs: List[float]) -> float:
    """Population skewness g1 = m3 / m2^1.5."""
    n = len(xs)
    if n < 3:
        return 0.0
    m = _mean(xs)
    m2 = sum((x - m) ** 2 for x in xs) / n
    m3 = sum((x - m) ** 3 for x in xs) / n
    return m3 / (m2 ** 1.5) if m2 > 1e-24 else 0.0


def _kurt(xs: List[float]) -> float:
    """Population excess kurtosis g2 = m4/m2^2 - 3."""
    n = len(xs)
    if n < 4:
        return 0.0
    m = _mean(xs)
    m2 = sum((x - m) ** 2 for x in xs) / n
    m4 = sum((x - m) ** 4 for x in xs) / n
    return m4 / (m2 ** 2) - 3.0 if m2 > 1e-24 else 0.0


def _q(xs: List[float], p: float) -> float:
    """Linear-interpolated quantile (numpy default)."""
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    h = (len(ys) - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, len(ys) - 1)
    frac = h - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def _cov(xs: List[float]) -> float:
    m = _mean(xs)
    return (_std(xs) / m) if m > 1e-12 else 0.0


def _zero_ratio(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(1 for x in xs if x <= 1e-9) / len(xs)


def _longest_zero_run(xs: List[float]) -> int:
    best = run = 0
    for x in xs:
        if x <= 1e-9:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _slope(xs: List[float]) -> float:
    """OLS slope vs day index."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = _mean(xs)
    num = sum((i - mx) * (y - my) for i, y in enumerate(xs))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den > 1e-12 else 0.0


def _mean_abs_diff(a: List[float], b: List[float]) -> float:
    if not a:
        return 0.0
    return sum(abs(u - v) for u, v in zip(a, b)) / len(a)


def _mean_abs_pct_change(a: List[float], b: List[float]) -> float:
    if not a:
        return 0.0
    vals = [abs(v - u) / abs(u) for u, v in zip(a, b) if abs(u) > 1e-9]
    return _mean(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def feature_names() -> List[str]:
    return [
        "mean_cons", "std_cons", "min_cons", "q25_cons", "median_cons",
        "q75_cons", "max_cons", "cv_cons",
        "skew_cons", "kurt_cons",
        "zero_ratio", "long_zero_run",
        "weekday_mean", "weekend_mean", "weekend_ratio",
        "month_spread",
        "half_ratio", "trend_slope",
        "mad_daily", "std_diff", "pct_change_mean",
        "zero_run_norm",
    ]


def feature_vector(series: List[Optional[float]], dow0: int = DEFAULT_DOW0) -> List[float]:
    x = impute(series)
    n = len(x)
    if n == 0:
        return [0.0] * len(feature_names())

    mu = _mean(x)
    sd = _std(x)

    wd = [x[i] for i in range(n) if ((dow0 + i) % 7) not in (0, 6)]
    we = [x[i] for i in range(n) if ((dow0 + i) % 7) in (0, 6)]
    wd_mu = _mean(wd)
    we_mu = _mean(we)
    if wd_mu > 1e-12:
        we_ratio = we_mu / wd_mu
    else:
        we_ratio = 1.0 if we_mu <= 1e-12 else 2.0

    bs = MONTH_BUCKET
    buckets = [x[i * bs:(i + 1) * bs] for i in range((n + bs - 1) // bs)]
    bucket_means = [_mean(b) for b in buckets if b]
    if len(bucket_means) >= 2 and mu > 1e-12:
        month_spread = (max(bucket_means) - min(bucket_means)) / mu
    else:
        month_spread = 0.0

    h = n // 2
    m1 = _mean(x[:h])
    m2 = _mean(x[h:]) if h > 0 else 0.0
    if m1 > 1e-12:
        half_ratio = m2 / m1
    else:
        half_ratio = 1.0 if m2 <= 1e-12 else 2.0

    if n > 1:
        prev = x[:-1]
        nxt = x[1:]
        diffs = [b - a for a, b in zip(prev, nxt)]
    else:
        prev = nxt = diffs = [0.0]

    lzr = _longest_zero_run(x)

    return [
        mu, sd, min(x), _q(x, 0.25), _q(x, 0.5), _q(x, 0.75), max(x),
        _cov(x),
        _skew(x), _kurt(x),
        _zero_ratio(x), float(lzr),
        wd_mu, we_mu, we_ratio,
        month_spread,
        half_ratio, _slope(x),
        _mean_abs_diff(prev, nxt), _std(diffs), _mean_abs_pct_change(prev, nxt),
        lzr / n,
    ]


def features_from_rows(rows: List[List[Optional[float]]], dow0: int = DEFAULT_DOW0) -> List[List[float]]:
    return [feature_vector(r, dow0=dow0) for r in rows]
