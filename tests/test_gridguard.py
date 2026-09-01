"""Pytest suite for GridGuard.

Covers: imputer semantics, feature invariants, vectorized-vs-reference
parity, dataset layout, and (when artifacts exist) ONNX-vs-sklearn parity
plus behavioral sanity on the committed fixture.

Run:  pytest -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIXTURE_CSV, MODELS_DIR, RAW_FULL
from src.features import (
    DEFAULT_DOW0, FEATURE_SPEC_VERSION, feature_names, feature_vector, impute,
)
from src.features_np import compute


# ---------------------------------------------------------------- imputer
def test_impute_all_missing():
    assert impute([None] * 10) == [0.0] * 10


def test_impute_clip_negatives():
    assert impute([-3.0, 2.0]) == [0.0, 2.0]


def test_impute_linear_interior():
    assert impute([1.0, None, None, 4.0]) == [1.0, 2.0, 3.0, 4.0]


def test_impute_edges_nearest():
    assert impute([None, None, 3.0, 6.0]) == [3.0, 3.0, 3.0, 6.0]
    assert impute([2.0, 4.0, None]) == [2.0, 4.0, 4.0]


def test_impute_empty():
    assert impute([]) == []


# ---------------------------------------------------------------- features
def test_feature_vector_length_and_finite():
    assert len(feature_names()) == 22
    v = feature_vector([1.0, 2.0, None, 4.0] * 30)
    assert len(v) == 22
    assert all(np.isfinite(v))


def test_flat_series_is_calm():
    v = dict(zip(feature_names(), feature_vector([7.0] * 90)))
    assert v["std_cons"] == 0.0
    assert v["cv_cons"] == 0.0
    assert v["zero_ratio"] == 0.0
    assert abs(v["weekend_ratio"] - 1.0) < 1e-9
    assert abs(v["half_ratio"] - 1.0) < 1e-9
    assert v["mad_daily"] == 0.0


def test_zero_heavy_series_flags_zero_features():
    v = dict(zip(feature_names(), feature_vector([0.0] * 45 + [3.0] * 45)))
    assert v["zero_ratio"] == pytest.approx(0.5, abs=1e-9)
    assert v["long_zero_run"] == 45
    assert v["zero_run_norm"] == pytest.approx(0.5, abs=1e-9)


def test_ramp_has_positive_slope():
    v = dict(zip(feature_names(), feature_vector([float(i) for i in range(90)])))
    assert v["trend_slope"] > 0.4
    assert v["half_ratio"] > 1.0


def test_all_zero_series_finite():
    v = dict(zip(feature_names(), feature_vector([0.0] * 90)))
    assert v["mean_cons"] == 0.0
    assert v["zero_ratio"] == 1.0
    assert all(np.isfinite(list(v.values())))


def test_quartiles_ordered():
    v = dict(zip(feature_names(),
                 feature_vector(list(np.random.RandomState(0).rand(120) * 10))))
    assert v["min_cons"] <= v["q25_cons"] <= v["median_cons"] \
        <= v["q75_cons"] <= v["max_cons"]


def test_dow0_changes_weekend_split():
    s = [10.0 if (i % 7) in (5, 6) else 2.0 for i in range(91)]
    a = dict(zip(feature_names(), feature_vector(s, dow0=0)))
    b = dict(zip(feature_names(), feature_vector(s, dow0=3)))
    assert a["weekend_mean"] > b["weekend_mean"]


# ---------------------------------------------------------------- parity
def test_np_vs_reference_parity():
    rng = np.random.RandomState(42)
    for trial in range(8):
        n = rng.randint(8, 200)
        raw = rng.rand(n) * 20
        holes = rng.rand(n) < rng.rand() * 0.5
        series = [None if h else float(x) for x, h in zip(raw, holes)]
        if all(v is None for v in series):
            continue
        X = np.array([impute(series)])
        F_np = compute(X, dow0=DEFAULT_DOW0)[0]
        F_ref = np.array(feature_vector(series, dow0=DEFAULT_DOW0))
        delta = np.abs(F_np - F_ref)
        assert np.allclose(F_np, F_ref, atol=1e-9, rtol=0), (
            f"trial {trial}: max delta {delta.max():.3e} at "
            f"{feature_names()[int(np.argmax(delta))]}")


def test_np_matrix_shape():
    X = np.array([[1.0, 2.0, 3.0, 4.0]] * 3)
    F = compute(X, dow0=3)
    assert F.shape == (3, 22)
    assert np.allclose(F[0], F[1])


def test_parity_goldens_exist():
    p = Path(__file__).parent / "data" / "parity_cases.json"
    if not p.exists():
        pytest.skip("run scripts/prepare.py first")
    data = json.loads(p.read_text())
    assert data["version"] == FEATURE_SPEC_VERSION
    assert data["n_features"] == 22
    for c in data["cases"]:
        assert len(c["features"]) == 22
        assert all(np.isfinite(c["features"]))


# ---------------------------------------------------------------- fixture
def test_fixture_exists_and_loads():
    if not FIXTURE_CSV.exists():
        pytest.skip("fixture not built yet — run scripts/prepare.py")
    import pandas as pd
    df = pd.read_csv(FIXTURE_CSV)
    assert len(df) == 24
    assert set(df["FLAG"].unique()) == {0, 1}
    lens = df["SERIES"].str.split(";").str.len()
    assert (lens == 90).all()
    # window anchored at day 944 = 2016-08-02, a Tuesday (JS getDay 2)
    assert (df["DOW0"] == 2).all()


# ---------------------------------------------------------------- raw data
@pytest.mark.skipif(not RAW_FULL.exists(), reason="raw SGCC csv not present")
def test_raw_sgcc_layout():
    from src.data import day_columns, load_sgcc
    df = load_sgcc()
    days = day_columns(df)
    assert len(df) == 42372
    assert len(days) == 1034
    assert int(df["FLAG"].sum()) == 3615
    assert str(days[0].date()) == "2014-01-01"
    assert (days[0].weekday() + 1) % 7 == 3  # Wednesday -> JS getDay 3


# ---------------------------------------------------------------- model
@pytest.mark.skipif(not (MODELS_DIR / "model.onnx").exists(),
                    reason="model not trained yet — run scripts/train.py")
def test_onnx_parity_and_behavior():
    """Deployed ONNX must reproduce sklearn scores on fixture consumers
    (raw series -> reference feature path), and flagged consumers must
    score higher on average."""
    import joblib
    import onnxruntime as ort
    import pandas as pd

    meta = json.loads((MODELS_DIR / "model_meta.json").read_text())
    est = joblib.load(MODELS_DIR / "final.joblib")

    df = pd.read_csv(FIXTURE_CSV)
    feats, flags = [], []
    for _, row in df.iterrows():
        series = [None if v == "" else float(v)
                  for v in row["SERIES"].split(";")]
        feats.append(feature_vector(series, dow0=int(row["DOW0"])))
        flags.append(int(row["FLAG"]))
    X = np.array(feats, dtype=np.float64)
    # sklearn = training-side scorer; ONNX = deployed scorer (float32 inputs)
    p_sk = est.predict_proba(X)[:, 1]
    X32 = X.astype(np.float32)

    sess = ort.InferenceSession(str(MODELS_DIR / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    outs = sess.run(None, {inp: X32})
    p_onnx = outs[1][:, 1] if len(outs) > 1 else outs[0]

    # converter fidelity: sklearn fed the SAME float32-rounded inputs
    p_sk32 = est.predict_proba(X32.astype(np.float64))[:, 1]
    conv_med = float(np.median(np.abs(p_onnx - p_sk32)))
    conv_max = float(np.abs(p_onnx - p_sk32).max())
    # float32 tree inference legitimately diverges near bin edges
    assert conv_med < 0.01, f"sklearn/ONNX median drift: {conv_med}"
    assert conv_max < 0.35, f"sklearn/ONNX tail drift: {conv_max}"
    # ranks stay aligned (float64 vs float32 boundary flips are expected)
    from scipy.stats import spearmanr
    rho = float(spearmanr(p_sk, p_onnx).statistic)
    assert rho > 0.9, f"sklearn/ONNX rank divergence: {rho}"

    flags = np.array(flags)
    assert p_onnx[flags == 1].mean() > p_onnx[flags == 0].mean(), \
        "model must score flagged (theft) consumers higher on average"

    t_hi = meta["thresholds"]["high"]
    t_med = meta["thresholds"]["balanced"]
    assert 0.0 < t_med < t_hi <= 1.0
    assert meta["metrics"]["consumer_mean"]["pr_auc"] > 0.25
    assert meta["metrics"]["consumer_mean"]["lift_top1p"] > 3
