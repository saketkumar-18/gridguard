"""Train window-level theft classifier; ONNX is the deployed source of truth.

Anti-leakage design:
  - split BY CONSUMER (a consumer's windows never straddle train/test)
  - CV = StratifiedGroupKFold grouped by consumer
  - consumer-level metrics on the DEPLOYED scorer (ONNX float32), with the
    sklearn float64 path kept as a diagnostic (rank-correlation gate)

Run:  python scripts/train.py
Out:  models/model.onnx, model_meta.json, final.joblib
      web/public/model/*, web/src/data/sample_pack.json
      reports/{metrics.json, pr_curve.png, roc_curve.png, feature_importance.png}
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import onnxmltools  # noqa: F401
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CV_FOLDS, FEATURES_CACHE, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE,
    WEB_DATA, WEB_PUBLIC_MODEL,
)
from src.features import feature_names

PRECISION_TARGET = 0.35
WINDOW = 90


def load_cache():
    df = pd.read_csv(FEATURES_CACHE)
    feats = feature_names()
    X = df[feats].astype(np.float64).values
    y = df["FLAG"].astype(int).values
    groups = df["CONS_NO"].astype(str).values
    return df, X, y, groups


def candidates() -> dict:
    # Scaler wrap keeps every ONNX tree input in a float32-safe range
    # (raw kurt/skew can hit 1e6+, where float32 rounding flips leaves).
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       C=1.0, random_state=RANDOM_STATE)),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "lgbm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=63,
                min_child_samples=25, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, class_weight="balanced", n_jobs=-1,
                max_bin=63,  # coarse bins -> float32-safe decisions in ONNX
                random_state=RANDOM_STATE, verbose=-1)),
        ]),
    }


def oof_proba(est, X, y, groups) -> np.ndarray:
    """Out-of-fold probabilities, grouped CV (no consumer crosses folds)."""
    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    p = np.zeros(len(y), dtype=np.float64)
    for tr, va in sgkf.split(X, y, groups):
        m = clone(est)
        m.fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]
    return p


def precision_target_threshold(y, proba, target=PRECISION_TARGET):
    prec, rec, thr = precision_recall_curve(y, proba)
    ok = np.where(prec[:-1] >= target)[0]
    if len(ok) == 0:
        return 0.95, 0.0, 0.0
    i = ok[np.argmax(rec[:-1][ok])]
    return float(thr[i]), float(prec[i]), float(rec[i])


def export_onnx(est, n_features: int, path: Path):
    from lightgbm import LGBMClassifier as _LGBMC
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import (
        calculate_linear_classifier_output_shapes,
    )
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
        convert_lightgbm,
    )
    try:
        update_registered_converter(
            _LGBMC, "LightGbmLGBMClassifier",
            calculate_linear_classifier_output_shapes, convert_lightgbm,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
    except Exception:
        pass
    onx = convert_sklearn(
        est,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
        options={"zipmap": False, "nocl": True},
        target_opset={"": 17, "ai.onnx.ml": 3},
    )
    path.write_bytes(onx.SerializeToString())


def ort_score(sess, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Score with the DEPLOYED artifact (ONNX float32) — the browser path."""
    inp = sess.get_inputs()[0].name
    out: list[np.ndarray] = []
    for s in range(0, len(X), chunk):
        outs = sess.run(None, {inp: X[s:s + chunk].astype(np.float32)})
        out.append(outs[1][:, 1] if len(outs) > 1 else outs[0])
    return np.concatenate(out)


def consumer_metrics(y_true, p_consumer, t_med):
    auc = roc_auc_score(y_true, p_consumer)
    prauc = average_precision_score(y_true, p_consumer)
    pred = (p_consumer >= t_med).astype(int)
    f1 = f1_score(y_true, pred)
    order = np.argsort(-p_consumer)
    n = len(y_true)
    p_top1 = float(y_true[order[:max(1, int(0.01 * n))]].mean())
    p_top5 = float(y_true[order[:max(1, int(0.05 * n))]].mean())
    return {
        "roc_auc": round(float(auc), 4), "pr_auc": round(float(prauc), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "top1p_precision": round(p_top1, 4), "top5p_precision": round(p_top5, 4),
        "base_rate": round(float(y_true.mean()), 4),
        "lift_top1p": round(p_top1 / float(y_true.mean()), 2),
        "n": int(n), "n_theft": int(y_true.sum()),
        "cm": confusion_matrix(y_true, pred).tolist(),
    }


def main() -> None:
    for d in (MODELS_DIR, REPORTS_DIR, WEB_PUBLIC_MODEL, WEB_DATA):
        d.mkdir(parents=True, exist_ok=True)

    df, X, y, groups = load_cache()
    print(f"window rows={X.shape[0]} features={X.shape[1]} "
          f"positives={int(y.sum())} ({100*y.mean():.2f}%)")

    # ---- consumer-level split ---------------------------------------------
    cons = pd.DataFrame({"cons": groups, "flag": y}).drop_duplicates("cons")
    tr_cons, te_cons = train_test_split(
        cons["cons"].values, test_size=TEST_SIZE,
        stratify=cons["flag"].values, random_state=RANDOM_STATE)
    tr_mask = np.isin(groups, tr_cons)
    te_mask = ~tr_mask
    X_tr, y_tr, g_tr = X[tr_mask], y[tr_mask], groups[tr_mask]
    X_te, y_te, g_te = X[te_mask], y[te_mask], groups[te_mask]
    print(f"consumers: train={len(tr_cons)} test={len(te_cons)} "
          f"window rows: train={tr_mask.sum()} test={te_mask.sum()}")

    # ---- CV model selection (cached across retries) -----------------------
    import os
    state_path = MODELS_DIR / "cv_state.joblib"
    cv_tab, oof_cache = [], {}
    if state_path.exists() and os.environ.get("GG_REUSE_CV"):
        st = joblib.load(state_path)
        cv_tab, oof_cache = st["cv_tab"], st["oof"]
        print("(reusing cached grouped-CV oof predictions)")
    if not cv_tab:
        print(f"\n{CV_FOLDS}-fold grouped CV (PR-AUC decides):")
        for name, est in candidates().items():
            p = oof_proba(est, X_tr, y_tr, g_tr)
            oof_cache[name] = p
            row = {"model": name,
                   "cv_roc_auc": round(float(roc_auc_score(y_tr, p)), 4),
                   "cv_pr_auc": round(float(average_precision_score(y_tr, p)), 4)}
            cv_tab.append(row)
            print(f"  {name:8s} ROC-AUC={row['cv_roc_auc']:.4f} PR-AUC={row['cv_pr_auc']:.4f}")
        joblib.dump({"cv_tab": cv_tab, "oof": oof_cache}, state_path)
    cv_tab = sorted(cv_tab, key=lambda r: -r["cv_pr_auc"])
    best_name = cv_tab[0]["model"]
    print(f"-> selected: {best_name}")

    # ---- threshold policy from out-of-fold train predictions ---------------
    tr_oof = oof_cache[best_name]
    t_high, p_hi, r_hi = precision_target_threshold(y_tr, tr_oof)
    prec, rec, thr = precision_recall_curve(y_tr, tr_oof)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    t_med = float(thr[int(np.nanargmax(f1s[:-1]))])
    print(f"thresholds (window-level): t_high(P>={PRECISION_TARGET})={t_high:.4f} "
          f"(P={p_hi:.3f} R={r_hi:.3f})  t_med(bestF1)={t_med:.4f}")

    # ---- final fit ---------------------------------------------------------
    print("final fit on train windows ...")
    final_est = candidates()[best_name]
    final_est.fit(X_tr, y_tr)
    joblib.dump(final_est, MODELS_DIR / "final.joblib")

    # ---- ONNX export = deployed scorer -------------------------------------
    print("\nONNX export ...")
    onnx_path = MODELS_DIR / "model.onnx"
    export_onnx(final_est, X.shape[1], onnx_path)
    print(f"  -> {onnx_path} ({onnx_path.stat().st_size/1e6:.2f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # sklearn float64 path — diagnostic only
    p_sk = final_est.predict_proba(X_te)[:, 1]
    # deployed float32 path — everything reported below uses THIS
    p_win = ort_score(sess, X_te)
    # sklearn fed the SAME float32-rounded inputs: isolates converter fidelity
    # from unavoidable float64->float32 input rounding on extreme features
    p_sk32 = final_est.predict_proba(X_te.astype(np.float32).astype(np.float64))[:, 1]

    rho = float(spearmanr(p_sk, p_win).statistic)
    max_delta = float(np.abs(p_sk - p_win).max())
    conv_delta = float(np.abs(p_win - p_sk32).max())
    conv_med = float(np.median(np.abs(p_win - p_sk32)))
    conv_p999 = float(np.quantile(np.abs(p_win - p_sk32), 0.999))
    auc_sk = roc_auc_score(y_te, p_sk)
    auc_onnx = roc_auc_score(y_te, p_win)
    print(f"  sklearn-vs-ONNX: spearman={rho:.5f} max|dp|={max_delta:.4f} "
          f"AUC sk={auc_sk:.4f} onnx={auc_onnx:.4f}")
    print(f"  float32 inference noise (same f32 inputs): median={conv_med:.2e} "
          f"p99.9={conv_p999:.4f} max={conv_delta:.4f}")
    # Deployed artifact is ONNX float32. Trees evaluated in float32 (scaler
    # arithmetic, thresholds, leaf accumulation) legitimately diverge from
    # float64 sklearn near bin edges. Gates: ranks aligned, AUC agrees, and
    # the noise tail is bounded (not a systematic converter error).
    assert conv_med < 0.005, f"converter median noise: {conv_med}"
    assert conv_p999 < 0.25, f"converter noise tail: {conv_p999}"
    assert rho > 0.98, f"rank divergence: {rho}"
    assert abs(auc_sk - auc_onnx) < 0.005, f"AUC divergence: {auc_sk} vs {auc_onnx}"

    win_prauc = average_precision_score(y_te, p_win)
    print(f"\nwindow-level TEST (ONNX scorer): ROC-AUC={auc_onnx:.4f} PR-AUC={win_prauc:.4f}")

    # ---- consumer-level aggregation (deployed scorer) ----------------------
    agg = pd.DataFrame({"cons": g_te, "flag": y_te, "p": p_win}) \
        .groupby("cons").agg(flag=("flag", "first"), p_mean=("p", "mean"),
                             p_max=("p", "max")).reset_index()
    agg_mean = consumer_metrics(agg["flag"].values, agg["p_mean"].values, t_med)
    print("consumer-level TEST (mean aggregation, ONNX):")
    print(f"  ROC-AUC={agg_mean['roc_auc']} PR-AUC={agg_mean['pr_auc']} "
          f"F1={agg_mean['f1']} P={agg_mean['precision']} R={agg_mean['recall']}")
    print(f"  top1%={agg_mean['top1p_precision']} top5%={agg_mean['top5p_precision']} "
          f"base={agg_mean['base_rate']} lift@1%={agg_mean['lift_top1p']}x")
    agg_max = consumer_metrics(agg["flag"].values, agg["p_max"].values, t_med)
    print(f"(max aggregation: PR-AUC={agg_max['pr_auc']}, lift@1%={agg_max['lift_top1p']}x)")

    # ---- importances / honest bands (raw feature space) --------------------
    clf = final_est[-1] if isinstance(final_est, Pipeline) else final_est
    try:
        imp = np.asarray(clf.feature_importances_, dtype=float)
    except AttributeError:
        imp = np.abs(np.asarray(clf.coef_[0], dtype=float))
    imp = imp / imp.sum()
    hm = y_tr == 0
    med = np.median(X_tr[hm], axis=0)
    q25 = np.quantile(X_tr[hm], 0.25, axis=0)
    q75 = np.quantile(X_tr[hm], 0.75, axis=0)

    meta = {
        "model": best_name,
        "n_features": int(X.shape[1]),
        "feature_names": feature_names(),
        "window_days": WINDOW,
        "scorer": "onnx-float32 (deployed)",
        "thresholds": {"high": round(t_high, 6), "balanced": round(t_med, 6)},
        "metrics": {
            "window": {"roc_auc": round(float(auc_onnx), 4),
                       "pr_auc": round(float(win_prauc), 4)},
            "consumer_mean": agg_mean,
            "consumer_max": agg_max,
            "sklearn_diag": {"roc_auc": round(float(auc_sk), 4),
                             "spearman_vs_onnx": round(rho, 5),
                             "f32_noise_median": round(conv_med, 6),
                             "f32_noise_p999": round(conv_p999, 4),
                             "f32_noise_max": round(conv_delta, 4)},
        },
        "cv_table": cv_tab,
        "importances": {n: round(float(v), 6) for n, v in zip(feature_names(), imp)},
        "honest_median": {n: round(float(v), 6) for n, v in zip(feature_names(), med)},
        "honest_q25": {n: round(float(v), 6) for n, v in zip(feature_names(), q25)},
        "honest_q75": {n: round(float(v), 6) for n, v in zip(feature_names(), q75)},
        "precision_target": PRECISION_TARGET,
    }
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copy2(onnx_path, WEB_PUBLIC_MODEL / "model.onnx")
    shutil.copy2(MODELS_DIR / "model_meta.json", WEB_PUBLIC_MODEL / "model_meta.json")

    # ---- sample pack for the web demo (deployed scorer) --------------------
    print("sample pack (recent window, ONNX-scored) ...")
    from src.data import load_sgcc, to_series_records
    from src.features import feature_vector as fv
    raw = load_sgcc()
    ids, flags, series, win_dow0 = to_series_records(raw, max_days=WINDOW)
    feats = np.array([fv(s, dow0=win_dow0) for s in series])
    scores = ort_score(sess, feats)
    rng = np.random.RandomState(RANDOM_STATE)
    flags_arr = np.asarray(flags)
    pos_idx = np.where(flags_arr == 1)[0]
    neg_idx = np.where(flags_arr == 0)[0]
    picks = list(rng.choice(pos_idx, 50, replace=False)) + \
            list(rng.choice(neg_idx, 50, replace=False))
    pack = [{
        "id": str(ids[i]), "flag": int(flags[i]),
        "score": round(float(scores[i]), 4),
        "series": [None if v is None else round(v, 2) for v in series[i]],
    } for i in picks]
    (WEB_DATA / "sample_pack.json").write_text(json.dumps(pack, separators=(",", ":")))
    print(f"  -> {WEB_DATA/'sample_pack.json'} ({len(pack)} consumers)")

    # ---- plots (deployed scorer) -------------------------------------------
    print("plots ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-darkgrid")

    y_c, p_c = agg["flag"].values, agg["p_mean"].values
    prec_t, rec_t, _ = precision_recall_curve(y_c, p_c)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    ax.plot(rec_t, prec_t, lw=2, color="#22d3ee",
            label=f"consumer-level (PR-AUC={agg_mean['pr_auc']})")
    ax.axhline(y_c.mean(), ls="--", c="#94a3b8",
               label=f"base rate ({y_c.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — hold-out consumers (SGCC)")
    ax.legend(); fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pr_curve.png"); plt.close(fig)

    fpr, tpr, _ = roc_curve(y_c, p_c)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    ax.plot(fpr, tpr, lw=2, color="#f59e0b",
            label=f"consumer-level ROC-AUC={agg_mean['roc_auc']}")
    ax.plot([0, 1], [0, 1], ls="--", c="#94a3b8")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — hold-out consumers (SGCC)")
    ax.legend(); fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png"); plt.close(fig)

    top = sorted(zip(feature_names(), imp), key=lambda t: -t[1])[:12]
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=130)
    ax.barh([t[0] for t in top][::-1], [t[1] for t in top][::-1], color="#34d399")
    ax.set_title("Top features by importance")
    ax.set_xlabel("importance share")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png"); plt.close(fig)

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(meta["metrics"], indent=2))
    print("\ntrain: DONE —", best_name,
          f"consumer PR-AUC={agg_mean['pr_auc']} lift@1%={agg_mean['lift_top1p']}x")


if __name__ == "__main__":
    main()
