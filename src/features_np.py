"""Vectorized feature computation — numerically mirrors src/features.py.

The pure-Python `features.feature_vector` is the cross-language reference
(TS parity target). This module computes the SAME 22 features with
numpy/pandas over the whole consumer matrix for training speed; tests
assert agreement within 1e-9.

Matrix layout: X[i, t] = consumption of consumer i on day t (imputed, no NaN).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .features import DEFAULT_DOW0, MONTH_BUCKET, feature_names

_EPS = 1e-9


def impute_days_matrix(days: pd.DataFrame) -> np.ndarray:
    """Impute a DataFrame of daily readings (columns = Timestamps).

    Spec (identical to features.impute):
      1. clip negatives -> 0
      2. linear interpolation over interior gaps (valid anchors on both sides)
      3. leading/trailing gaps <- nearest valid value
      4. all-missing consumer -> 0
    """
    X = days.clip(lower=0.0)
    X = X.interpolate(method="linear", axis=1, limit_area="inside")
    X = X.ffill(axis=1).bfill(axis=1)
    X = X.fillna(0.0)
    return X.to_numpy(dtype=np.float64)


def _zero_run_max(M: np.ndarray) -> np.ndarray:
    """Longest run of (value <= eps) along axis 1, vectorized over rows."""
    n_rows = M.shape[0]
    run = np.zeros(n_rows, dtype=np.int64)
    best = np.zeros(n_rows, dtype=np.int64)
    for k in range(M.shape[1]):
        hit = M[:, k] <= _EPS
        run = np.where(hit, run + 1, 0)
        np.maximum(best, run, out=best)
    return best


def compute(X: np.ndarray, dow0: int = DEFAULT_DOW0) -> np.ndarray:
    """22-feature matrix for all consumers. Order = features.feature_names()."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, T = X.shape
    out = np.zeros((n, len(feature_names())), dtype=np.float64)

    if T == 0:
        return out

    mu = X.mean(axis=1)
    if T > 1:
        sd = X.std(axis=1, ddof=1)
    else:
        sd = np.zeros(n)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    q25, med, q75 = (np.quantile(X, p, axis=1) for p in (0.25, 0.5, 0.75))
    cv = np.divide(sd, mu, out=np.zeros(n), where=mu > 1e-12)

    # population moments for skew/kurtosis (mirrors features._skew/_kurt)
    if T >= 3:
        d = X - mu[:, None]
        m2 = (d**2).mean(axis=1)
        m3 = (d**3).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = np.where(m2 > 1e-24, m3 / np.sqrt(m2**3), 0.0)
    else:
        skew = np.zeros(n)
    if T >= 4:
        d = X - mu[:, None]
        m2 = (d**2).mean(axis=1)
        m4 = (d**4).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            kurt = np.where(m2 > 1e-24, m4 / (m2**2) - 3.0, 0.0)
    else:
        kurt = np.zeros(n)

    zr = (X <= _EPS).mean(axis=1)
    lzr = _zero_run_max(X).astype(np.float64)

    # weekday / weekend (dow0 = JS getDay of day 0)
    dow = (dow0 + np.arange(T)) % 7
    we_mask = np.isin(dow, [0, 6])
    wd_mask = ~we_mask
    if wd_mask.any():
        wd_mu = np.where(wd_mask, X, 0.0).sum(axis=1) / np.maximum(wd_mask.sum(), 1)
    else:
        wd_mu = np.zeros(n)
    if we_mask.any():
        we_mu = np.where(we_mask, X, 0.0).sum(axis=1) / np.maximum(we_mask.sum(), 1)
    else:
        we_mu = np.zeros(n)
    we_ratio = np.where(wd_mu > 1e-12, we_mu / np.where(wd_mu > 1e-12, wd_mu, 1.0),
                       np.where(we_mu <= 1e-12, 1.0, 2.0))

    # month spread: means of consecutive 30-day buckets, (max-min)/mean
    nb = (T + MONTH_BUCKET - 1) // MONTH_BUCKET
    bucket_means = np.zeros((n, nb))
    for b in range(nb):
        seg = X[:, b * MONTH_BUCKET:(b + 1) * MONTH_BUCKET]
        bucket_means[:, b] = seg.mean(axis=1) if seg.shape[1] else 0.0
    if nb >= 2:
        ms = (bucket_means.max(axis=1) - bucket_means.min(axis=1))
        month_spread = np.divide(ms, mu, out=np.zeros(n), where=mu > 1e-12)
    else:
        month_spread = np.zeros(n)

    # halves
    h = T // 2
    m1 = X[:, :h].mean(axis=1) if h > 0 else np.zeros(n)
    m2h = X[:, h:].mean(axis=1) if T - h > 0 else np.zeros(n)
    if h > 0:
        half_ratio = np.where(m1 > 1e-12, m2h / np.where(m1 > 1e-12, m1, 1.0),
                              np.where(m2h <= 1e-12, 1.0, 2.0))
    else:
        half_ratio = np.ones(n)

    # OLS slope vs day index
    if T > 1:
        idx = np.arange(T, dtype=np.float64)
        mx_i = (T - 1) / 2.0
        num = ((idx - mx_i)[None, :] * X).sum(axis=1) - X.mean(axis=1) * ((idx - mx_i).sum())
        den = float(((idx - mx_i) ** 2).sum())
        slope = num / den
    else:
        slope = np.zeros(n)

    # daily diffs
    if T > 1:
        prev = X[:, :-1]
        nxt = X[:, 1:]
        diffs = nxt - prev
        mad = np.abs(nxt - prev).mean(axis=1)
        std_diff = diffs.std(axis=1, ddof=1) if T > 2 else np.zeros(n)
        denom_ok = np.abs(prev) > 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(denom_ok, np.abs(nxt - prev) / np.where(denom_ok, np.abs(prev), 1.0), 0.0)
        cnt = denom_ok.sum(axis=1)
        pct_mean = np.divide(pct.sum(axis=1), cnt, out=np.zeros(n), where=cnt > 0)
    else:
        mad = std_diff = pct_mean = np.zeros(n)

    cols = [mu, sd, mn, q25, med, q75, mx, cv, skew, kurt, zr, lzr,
            wd_mu, we_mu, we_ratio, month_spread, half_ratio, slope,
            mad, std_diff, pct_mean, lzr / T]
    for j, c in enumerate(cols):
        out[:, j] = c
    return out
