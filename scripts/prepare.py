"""Build windowed feature cache, fixtures, JS-parity goldens, EDA.

Design: each consumer's 1034-day series is cut into 11 windows of 90 days
(10 non-overlapping + 1 end-anchored "recent" window) so that training
mirrors deployment, where a model scores a consumer's RECENT window.
Imputation runs on the FULL series first, then windows are sliced from the
imputed matrix (edge gaps borrow from outside the window — honest).

Run:  python scripts/prepare.py
Out:  data/processed/features_full.csv  (42372 x 11 windows x 22 features)
      tests/data/fixture.csv            (24 consumers, recent 90-day window, raw)
      tests/data/parity_cases.json      (JS<->Python goldens)
      reports/eda.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import (
    FEATURES_CACHE, FIXTURE_CSV, PROCESSED_DIR, REPORTS_DIR, RANDOM_STATE,
)
from src.data import day_columns, load_sgcc, to_series_records
from src.features import feature_names, feature_vector
from src.features_np import compute, impute_days_matrix

PARITY_CASES = Path(__file__).resolve().parents[1] / "web" / "src" / "data" / "parity_goldens.json"
PY_PARITY_CASES = Path(__file__).resolve().parents[1] / "tests" / "data" / "parity_cases.json"
WINDOW = 90


def window_starts(T: int, w: int = WINDOW) -> list[int]:
    starts = list(range(0, T - w + 1, w))
    last = T - w
    if last not in starts:
        starts.append(last)
    return starts


def main() -> None:
    print("loading raw SGCC ...")
    df = load_sgcc()
    days = day_columns(df)
    T = len(days)
    print(f"consumers={len(df)} days={T} theft={int(df['FLAG'].sum())} "
          f"({100*df['FLAG'].mean():.2f}%)")

    print("imputing full matrix (clip -> interp -> edges) ...")
    Xi = impute_days_matrix(df[days])
    assert not np.isnan(Xi).any()

    starts = window_starts(T)
    print(f"{len(starts)} windows/consumer of {WINDOW} days: starts={starts[:3]}..{starts[-1]}")

    print("computing 22 features per window (vectorized) ...")
    frames = []
    for wid, s in enumerate(starts):
        dow0 = (days[s].weekday() + 1) % 7
        F = compute(Xi[:, s:s + WINDOW], dow0=dow0)
        wdf = pd.DataFrame(F, columns=feature_names())
        wdf.insert(0, "WID", wid)
        wdf.insert(0, "FLAG", df["FLAG"].astype(int).values)
        wdf.insert(0, "CONS_NO", df.index)
        frames.append(wdf)
        print(f"  window {wid:2d} (day {s:4d}-{s+WINDOW-1}, dow0={dow0}) done")
    cache = pd.concat(frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache.to_csv(FEATURES_CACHE, index=False)
    print(f"features cache -> {FEATURES_CACHE}  shape={cache.shape}")

    # ---------------- fixture: 24 stratified consumers, recent window ------
    print("building fixture (recent 90-day window, RAW with gaps) ...")
    rng = random.Random(RANDOM_STATE)
    ids, flags, series, win_dow0 = to_series_records(df, max_days=WINDOW)
    last_wid = len(starts) - 1
    assert (T - WINDOW) == starts[last_wid], "fixture window == end-anchored window"
    idx_pos = [i for i, f in enumerate(flags) if f == 1]
    idx_neg = [i for i, f in enumerate(flags) if f == 0]
    rows = []
    for i in rng.sample(idx_pos, 12) + rng.sample(idx_neg, 12):
        rows.append({
            "CONS_NO": ids[i], "FLAG": flags[i], "WID": last_wid,
            "DOW0": win_dow0,
            "SERIES": ";".join("" if v is None else f"{v:.4g}" for v in series[i]),
        })
    FIXTURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(FIXTURE_CSV, index=False)
    print(f"fixture -> {FIXTURE_CSV} ({len(rows)} rows, dow0={win_dow0})")

    # ---------------- JS-parity goldens ------------------------------------
    print("building JS-parity goldens ...")
    cases = []
    synth = [
        {"dow0": 3, "series": [None] * 30},
        {"dow0": 3, "series": [5.0] * 60},
        {"dow0": 3, "series": [0.0] * 60},
        {"dow0": 3, "series": [float(i) for i in range(1, 61)]},
        {"dow0": 3, "series": [10.0, None, 8.0, None, None, 6.0] * 10},
        {"dow0": 3, "series": [3.0, -5.0, 2.0, 0.0, None, 7.0] * 15},
        {"dow0": 3, "series": [abs(20 * np.sin(i / 5)) for i in range(90)]},
        {"dow0": 0, "series": [abs(20 * np.sin(i / 5)) for i in range(90)]},
    ]
    for c in synth:
        cases.append({
            "dow0": c["dow0"],
            "series": c["series"],
            "features": feature_vector(c["series"], dow0=c["dow0"]),
        })
    for _ in range(9):
        i = rng.randrange(len(series))
        raw = [None if v is None else round(v, 4) for v in series[i]]
        cases.append({
            "dow0": win_dow0,
            "series": raw,
            "features": feature_vector(raw, dow0=win_dow0),
        })
    payload = json.dumps(
        {"version": 2, "n_features": len(feature_names()), "cases": cases},
        separators=(",", ":"))
    PARITY_CASES.write_text(payload)
    PY_PARITY_CASES.write_text(payload)
    print(f"parity goldens -> {PARITY_CASES} + {PY_PARITY_CASES} ({len(cases)} cases)")

    # ---------------- EDA --------------------------------------------------
    print("EDA ...")
    raw_days = df[days]
    eda = {
        "n_consumers": int(len(df)),
        "n_days": int(T),
        "n_theft": int(df["FLAG"].sum()),
        "theft_pct": round(100 * float(df["FLAG"].mean()), 3),
        "missing_pct_overall": round(100 * float(raw_days.isna().mean().mean()), 3),
        "first_day": str(days[0].date()),
        "last_day": str(days[-1].date()),
        "window_days": WINDOW,
        "n_windows": len(starts),
        "mean_daily_kwh_honest": round(float(raw_days[df.FLAG == 0].mean().mean()), 4),
        "mean_daily_kwh_theft": round(float(raw_days[df.FLAG == 1].mean().mean()), 4),
        "zero_readings_pct_honest": round(
            100 * float((raw_days[df.FLAG == 0] == 0).mean().mean()), 4),
        "zero_readings_pct_theft": round(
            100 * float((raw_days[df.FLAG == 1] == 0).mean().mean()), 4),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "eda.json").write_text(json.dumps(eda, indent=2))
    print(json.dumps(eda, indent=2))
    print("prepare: DONE")


if __name__ == "__main__":
    main()
