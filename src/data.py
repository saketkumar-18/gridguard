"""Data loading and cleaning for the SGCC smart-meter dataset.

Raw CSV layout: first ~1034 columns are daily consumption with M/D/YYYY
headers, then CONS_NO (consumer id) and FLAG (0 = honest, 1 = theft).
We load into a tidy frame indexed by CONS_NO with Timestamp day columns.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import pandas as pd

from .config import RAW_FULL, RAW_SMALL

_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")


def load_sgcc(path: Optional[str] = None, small: bool = False) -> pd.DataFrame:
    """Load raw SGCC csv -> DataFrame indexed by CONS_NO.

    Columns: FLAG (int) + one column per day (Timestamp), chronological order.
    Values: float kWh (NaN = missing), negatives kept as-is (imputer clips).
    """
    path = path or (str(RAW_SMALL) if small else str(RAW_FULL))
    df = pd.read_csv(path, low_memory=False)
    date_cols = [c for c in df.columns if _DATE_RE.match(str(c))]
    meta_cols = [c for c in df.columns if not _DATE_RE.match(str(c))]
    if "CONS_NO" not in meta_cols or "FLAG" not in meta_cols:
        raise ValueError(f"unexpected SGCC layout at {path}")
    out = pd.DataFrame(index=df["CONS_NO"].astype(str).str.strip().values)
    out["FLAG"] = df["FLAG"].astype(int).values
    dsub = df[date_cols].astype(float)
    dsub.index = out.index
    dsub.columns = pd.DatetimeIndex(pd.to_datetime(date_cols, format="mixed"))
    day_block = dsub[sorted(dsub.columns)]
    return pd.concat([out, day_block], axis=1)


def day_columns(df: pd.DataFrame) -> List[pd.Timestamp]:
    return sorted([c for c in df.columns if c != "FLAG"])


def to_series_records(
    df: pd.DataFrame, max_days: Optional[int] = None
) -> Tuple[List[str], List[int], List[List[Optional[float]]], int]:
    """Split into (ids, flags, raw series, dow0-of-window).

    If max_days is given, keep the LAST max_days days. NaN -> None.
    dow0 is the JS getDay() (0=Sunday) of the first day of the kept window.
    """
    days = day_columns(df)
    if max_days is not None and len(days) > max_days:
        days = days[-max_days:]
    dow0 = (days[0].weekday() + 1) % 7  # Mon=0 py -> 1; Sun=6 py -> 0
    sub = df[days]
    ids = [str(i) for i in df.index]
    flags = df["FLAG"].astype(int).tolist()
    series: List[List[Optional[float]]] = [
        [None if pd.isna(v) else float(v) for v in row]
        for row in sub.itertuples(index=False, name=None)
    ]
    return ids, flags, series, dow0
