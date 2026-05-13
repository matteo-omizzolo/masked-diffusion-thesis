"""Held-out state-predictability audit for corrector timing.

This script uses existing Phase 1 Protocol A trajectories. It does not run a
model, does not call the ProSeCo backend, and does not require HPC.

Question:
    Do observable state features before correction predict the single-step
    correction value delta_{i,t} better than time alone?

Outputs:
    results/state_predictability_<gitsha>/
      config.json
      prediction_rows.json
      aggregate_stats.json
      interpretation.md
      figures/*.png
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # Pillow is available locally; matplotlib is not required.
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional plotting fallback
    Image = None
    ImageDraw = None
    ImageFont = None


DEFAULT_INPUT_DIR = Path("results/phase1_proseco_owt_full/protocol_a")
DEFAULT_RESULTS_PREFIX = Path("results/state_predictability")

STATE_FEATURES = (
    "entropy",
    "inverse_margin",
    "quality_mass_proxy",
    "unmasked_fraction",
    "n_revisable",
    "n_masked",
)


@dataclass(frozen=True)
class Row:
    seed: int
    t: int
    T: int
    delta: float
    features: dict[str, float]

    @property
    def t_norm(self) -> float:
        if self.T <= 1:
            return 0.0
        return self.t / float(self.T - 1)


def git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def load_protocol_a_rows(input_dir: Path) -> list[Row]:
    files = sorted(input_dir.glob("trajectory_*.json"))
    if not files:
        raise FileNotFoundError(f"no trajectory_*.json files found in {input_dir}")
    rows: list[Row] = []
    for path in files:
        obj = json.loads(path.read_text())
        seed = int(obj["seed"])
        T = int(obj["T"])
        for step in obj["per_t"]:
            features: dict[str, float] = {}
            for name in STATE_FEATURES:
                if name not in step:
                    raise KeyError(f"{path}:{step.get('t')} missing {name}")
                features[name] = float(step[name])
            rows.append(
                Row(
                    seed=seed,
                    t=int(step["t"]),
                    T=T,
                    delta=float(step["delta"]),
                    features=features,
                )
            )
    return sorted(rows, key=lambda r: (r.seed, r.t))


def make_seed_folds(seeds: Sequence[int], n_folds: int, seed: int = 1729) -> list[list[int]]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    unique = sorted(set(int(s) for s in seeds))
    if n_folds > len(unique):
        raise ValueError("n_folds cannot exceed number of seeds")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique, dtype=int)
    rng.shuffle(shuffled)
    folds = [list(map(int, fold)) for fold in np.array_split(shuffled, n_folds)]
    if any(not fold for fold in folds):
        raise ValueError("empty fold produced")
    return folds


def _feature_vector(row: Row, kind: str) -> list[float]:
    t = row.t_norm
    time_feats = [t, t * t, t * t * t]
    state_feats = [row.features[name] for name in STATE_FEATURES]
    if kind == "time":
        return time_feats
    if kind == "state":
        return state_feats
    if kind == "time_state":
        return time_feats + state_feats
    raise ValueError(f"unknown feature kind {kind!r}")


def _matrix(rows: Sequence[Row], kind: str) -> np.ndarray:
    return np.asarray([_feature_vector(r, kind) for r in rows], dtype=np.float64)


def _target(rows: Sequence[Row]) -> np.ndarray:
    return np.asarray([r.delta for r in rows], dtype=np.float64)


@dataclass
class RidgeModel:
    kind: str
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def predict(self, rows: Sequence[Row]) -> np.ndarray:
        X = _matrix(rows, self.kind)
        Xs = (X - self.mean) / self.scale
        design = np.column_stack([np.ones(len(rows)), Xs])
        return design @ self.coef


def fit_ridge(rows: Sequence[Row], kind: str, alpha: float = 1.0) -> RidgeModel:
    X = _matrix(rows, kind)
    y = _target(rows)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Xs = (X - mean) / scale
    design = np.column_stack([np.ones(len(rows)), Xs])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RidgeModel(kind=kind, alpha=alpha, mean=mean, scale=scale, coef=coef)


def predict_global_mean(train: Sequence[Row], test: Sequence[Row]) -> np.ndarray:
    value = float(np.mean([r.delta for r in train]))
    return np.full(len(test), value, dtype=np.float64)


def predict_time_mean(train: Sequence[Row], test: Sequence[Row]) -> np.ndarray:
    by_t: dict[int, list[float]] = {}
    for row in train:
        by_t.setdefault(row.t, []).append(row.delta)
    fallback = float(np.mean([r.delta for r in train]))
    means = {t: float(np.mean(vals)) for t, vals in by_t.items()}
    return np.asarray([means.get(r.t, fallback) for r in test], dtype=np.float64)


def predict_phase_mean(train: Sequence[Row], test: Sequence[Row], n_bins: int = 4) -> np.ndarray:
    by_bin: dict[int, list[float]] = {}
    for row in train:
        b = min(n_bins - 1, int(row.t_norm * n_bins))
        by_bin.setdefault(b, []).append(row.delta)
    fallback = float(np.mean([r.delta for r in train]))
    means = {b: float(np.mean(vals)) for b, vals in by_bin.items()}
    out = []
    for row in test:
        b = min(n_bins - 1, int(row.t_norm * n_bins))
        out.append(means.get(b, fallback))
    return np.asarray(out, dtype=np.float64)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) < 2:
        return float("nan")
    rx = _rank_average(x_arr)
    ry = _rank_average(y_arr)
    sx = rx.std()
    sy = ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.mean(((rx - rx.mean()) / sx) * ((ry - ry.mean()) / sy)))


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def evaluate_predictions(rows: Sequence[dict[str, Any]], model_names: Sequence[str]) -> dict[str, Any]:
    y = np.asarray([float(r["delta"]) for r in rows], dtype=np.float64)
    out: dict[str, Any] = {}
    for model in model_names:
        pred = np.asarray([float(r[f"pred_{model}"]) for r in rows], dtype=np.float64)
        err = pred - y
        out[model] = {
            "mse": float(np.mean(err * err)),
            "mae": float(np.mean(np.abs(err))),
            "spearman_pred_delta": spearman(pred, y),
        }
    return out


def bootstrap_seed_improvement(
    prediction_rows: Sequence[dict[str, Any]],
    baseline: str,
    candidate: str,
    n_resamples: int = 2000,
    seed: int = 1729,
) -> dict[str, float]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    seeds = sorted(by_seed)
    diffs = []
    for s in seeds:
        rows = by_seed[s]
        y = np.asarray([float(r["delta"]) for r in rows])
        pb = np.asarray([float(r[f"pred_{baseline}"]) for r in rows])
        pc = np.asarray([float(r[f"pred_{candidate}"]) for r in rows])
        diffs.append(float(np.mean((pb - y) ** 2) - np.mean((pc - y) ** 2)))
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(diffs_arr), size=len(diffs_arr))
        boot.append(float(np.mean(diffs_arr[idx])))
    lo, hi = np.quantile(np.asarray(boot), [0.025, 0.975])
    return {
        "baseline": baseline,
        "candidate": candidate,
        "mean_mse_reduction": float(np.mean(diffs_arr)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "pct_seeds_improved": float(np.mean(diffs_arr > 0.0)),
    }


def uniform_schedule(T: int, B: int) -> list[int]:
    if B <= 0:
        return []
    if B >= T:
        return list(range(T))
    return sorted({int(i * T / B) for i in range(B)})


def additive_schedule_metrics(
    prediction_rows: Sequence[dict[str, Any]],
    model_names: Sequence[str],
    B_values: Sequence[int],
) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    out: dict[str, Any] = {}
    for B in B_values:
        per_model_gain: dict[str, list[float]] = {m: [] for m in model_names}
        per_model_close: dict[str, list[float]] = {m: [] for m in model_names}
        per_model_overlap: dict[str, list[float]] = {m: [] for m in model_names}
        uniform_gains: list[float] = []
        oracle_gains: list[float] = []
        for seed, rows in by_seed.items():
            ordered = sorted(rows, key=lambda r: int(r["t"]))
            T = len(ordered)
            deltas = np.asarray([float(r["delta"]) for r in ordered], dtype=np.float64)
            u_steps = uniform_schedule(T, B)
            uniform_gain = float(np.sum(deltas[u_steps]))
            oracle_steps = set(np.argsort(-deltas)[:B].astype(int).tolist())
            oracle_gain = float(np.sum([deltas[t] for t in oracle_steps]))
            denom = oracle_gain - uniform_gain
            uniform_gains.append(uniform_gain)
            oracle_gains.append(oracle_gain)
            for model in model_names:
                pred = np.asarray([float(r[f"pred_{model}"]) for r in ordered])
                steps = set(np.argsort(-pred)[:B].astype(int).tolist())
                gain = float(np.sum([deltas[t] for t in steps]))
                per_model_gain[model].append(gain)
                if abs(denom) > 1e-12:
                    per_model_close[model].append((gain - uniform_gain) / denom)
                overlap = len(steps & oracle_steps) / float(B)
                per_model_overlap[model].append(overlap)
        block: dict[str, Any] = {
            "B": B,
            "uniform_gain_mean": float(np.mean(uniform_gains)),
            "oracle_additive_gain_mean": float(np.mean(oracle_gains)),
            "models": {},
        }
        mean_denom = block["oracle_additive_gain_mean"] - block["uniform_gain_mean"]
        for model in model_names:
            gain_mean = float(np.mean(per_model_gain[model]))
            block["models"][model] = {
                "gain_mean": gain_mean,
                "close_ratio_mean_of_seeds": float(np.mean(per_model_close[model]))
                if per_model_close[model]
                else float("nan"),
                "close_ratio_from_means": float((gain_mean - block["uniform_gain_mean"]) / mean_denom)
                if abs(mean_denom) > 1e-12
                else float("nan"),
                "oracle_overlap_mean": float(np.mean(per_model_overlap[model])),
            }
        out[str(B)] = block
    return out


def run_audit(
    rows: Sequence[Row],
    *,
    n_folds: int = 5,
    fold_seed: int = 1729,
    ridge_alpha: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seeds = sorted({r.seed for r in rows})
    folds = make_seed_folds(seeds, n_folds=n_folds, seed=fold_seed)
    prediction_rows: list[dict[str, Any]] = []
    for fold_id, test_seeds in enumerate(folds):
        test_set = set(test_seeds)
        train = [r for r in rows if r.seed not in test_set]
        test = [r for r in rows if r.seed in test_set]
        ridge_time = fit_ridge(train, "time", alpha=ridge_alpha)
        ridge_state = fit_ridge(train, "state", alpha=ridge_alpha)
        ridge_time_state = fit_ridge(train, "time_state", alpha=ridge_alpha)
        preds = {
            "global_mean": predict_global_mean(train, test),
            "phase_mean": predict_phase_mean(train, test),
            "time_mean": predict_time_mean(train, test),
            "ridge_time": ridge_time.predict(test),
            "ridge_state": ridge_state.predict(test),
            "ridge_time_state": ridge_time_state.predict(test),
        }
        for idx, row in enumerate(test):
            out = {
                "seed": row.seed,
                "t": row.t,
                "T": row.T,
                "fold_id": fold_id,
                "delta": row.delta,
                "t_norm": row.t_norm,
            }
            for name in STATE_FEATURES:
                out[name] = row.features[name]
            for model_name, pred in preds.items():
                out[f"pred_{model_name}"] = float(pred[idx])
            prediction_rows.append(out)
    model_names = [
        "global_mean",
        "phase_mean",
        "time_mean",
        "ridge_time",
        "ridge_state",
        "ridge_time_state",
    ]
    predictive = evaluate_predictions(prediction_rows, model_names)
    improvements = {
        "ridge_time_state_vs_ridge_time": bootstrap_seed_improvement(
            prediction_rows, "ridge_time", "ridge_time_state"
        ),
        "ridge_time_state_vs_time_mean": bootstrap_seed_improvement(
            prediction_rows, "time_mean", "ridge_time_state"
        ),
        "ridge_state_vs_ridge_time": bootstrap_seed_improvement(
            prediction_rows, "ridge_time", "ridge_state"
        ),
    }
    schedule_metrics = additive_schedule_metrics_with_uniform(
        prediction_rows, model_names=model_names, B_values=[2, 3, 4]
    )
    verdict = make_verdict(predictive, improvements, schedule_metrics)
    aggregate = {
        "meta": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generated_by": f"scripts/proseco/state_predictability/analyze_state_predictability.py@{git_short_hash()}",
            "n_rows": len(prediction_rows),
            "n_seeds": len(seeds),
            "n_folds": n_folds,
            "fold_seed": fold_seed,
            "ridge_alpha": ridge_alpha,
            "state_features": list(STATE_FEATURES),
            "excluded_post_correction_fields": ["n_changed", "tcr", "f_branch"],
        },
        "predictive_metrics": predictive,
        "mse_improvements": improvements,
        "additive_schedule_metrics": schedule_metrics,
        "verdict": verdict,
    }
    return prediction_rows, aggregate


def additive_schedule_metrics_with_uniform(
    prediction_rows: Sequence[dict[str, Any]],
    model_names: Sequence[str],
    B_values: Sequence[int],
) -> dict[str, Any]:
    metrics = additive_schedule_metrics(prediction_rows, model_names, B_values)
    for block in metrics.values():
        block["models"]["uniform"] = {
            "gain_mean": block["uniform_gain_mean"],
            "close_ratio_mean_of_seeds": 0.0,
            "close_ratio_from_means": 0.0,
            "oracle_overlap_mean": float("nan"),
        }
    return metrics


def make_verdict(
    predictive: dict[str, Any],
    improvements: dict[str, Any],
    schedule_metrics: dict[str, Any],
) -> dict[str, Any]:
    state_vs_time = improvements["ridge_time_state_vs_ridge_time"]
    pct = state_vs_time["pct_seeds_improved"]
    ci_lo = state_vs_time["ci95_lo"]
    mse_reduction = state_vs_time["mean_mse_reduction"]
    close_b3 = schedule_metrics["3"]["models"]["ridge_time_state"]["close_ratio_from_means"]
    if ci_lo > 0 and pct >= 0.6 and close_b3 > 0.25:
        label = "state_predictability_supported"
    elif mse_reduction > 0 and pct >= 0.5:
        label = "state_predictability_weak"
    else:
        label = "state_predictability_not_supported"
    return {
        "label": label,
        "state_vs_time_mse_reduction": mse_reduction,
        "state_vs_time_ci95_lo": ci_lo,
        "state_vs_time_ci95_hi": state_vs_time["ci95_hi"],
        "pct_seeds_state_beats_time": pct,
        "ridge_time_state_spearman": predictive["ridge_time_state"]["spearman_pred_delta"],
        "ridge_time_spearman": predictive["ridge_time"]["spearman_pred_delta"],
        "B3_additive_close_ratio_from_means": close_b3,
        "interpretation": (
            "A full online scheduler is justified only if this label is "
            "state_predictability_supported. Weak/not-supported labels are "
            "evidence for writing the diagnostic result rather than launching "
            "a larger online-control phase."
        ),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def write_interpretation(out_dir: Path, aggregate: dict[str, Any]) -> None:
    verdict = aggregate["verdict"]
    pred = aggregate["predictive_metrics"]
    imp = aggregate["mse_improvements"]["ridge_time_state_vs_ridge_time"]
    sched = aggregate["additive_schedule_metrics"]
    lines = [
        "# State-Predictability Audit",
        "",
        f"Verdict: **{verdict['label']}**.",
        "",
        "## Main Readout",
        "",
        (
            "This audit asks whether observable pre-correction state features "
            "predict single-step correction value better than time alone."
        ),
        "",
        f"- Rows: {aggregate['meta']['n_rows']} seed-step observations.",
        f"- Seeds: {aggregate['meta']['n_seeds']}, grouped {aggregate['meta']['n_folds']}-fold holdout.",
        f"- Ridge time-only MSE: {pred['ridge_time']['mse']:.6f}.",
        f"- Ridge time+state MSE: {pred['ridge_time_state']['mse']:.6f}.",
        (
            "- Mean seed-level MSE reduction from adding state features: "
            f"{imp['mean_mse_reduction']:.6f} "
            f"(95% CI [{imp['ci95_lo']:.6f}, {imp['ci95_hi']:.6f}])."
        ),
        f"- Fraction of seeds improved by time+state over time-only: {imp['pct_seeds_improved']:.3f}.",
        f"- Spearman(pred, delta), time-only: {pred['ridge_time']['spearman_pred_delta']:.3f}.",
        f"- Spearman(pred, delta), time+state: {pred['ridge_time_state']['spearman_pred_delta']:.3f}.",
        "",
        "## Additive Top-B Timing",
        "",
    ]
    for B in (2, 3, 4):
        block = sched[str(B)]
        ts = block["models"]["ridge_time_state"]
        tm = block["models"]["ridge_time"]
        lines.extend(
            [
                (
                    f"- B={B}: time+state close ratio from means = "
                    f"{ts['close_ratio_from_means']:.3f}; "
                    f"time-only ridge = {tm['close_ratio_from_means']:.3f}; "
                    f"oracle additive gain = {block['oracle_additive_gain_mean']:.3f}; "
                    f"uniform gain = {block['uniform_gain_mean']:.3f}."
                )
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            verdict["interpretation"],
            "",
            "This result is ProSeCo-OWT-specific. The framework is model-agnostic, "
            "but the empirical state-predictability verdict is not.",
            "",
        ]
    )
    (out_dir / "interpretation.md").write_text("\n".join(lines))


def write_figures(out_dir: Path, aggregate: dict[str, Any], prediction_rows: Sequence[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if Image is None:
        return
    _plot_model_mse(fig_dir / "model_mse.png", aggregate["predictive_metrics"])
    _plot_close_ratios(fig_dir / "additive_close_ratios.png", aggregate["additive_schedule_metrics"])
    _plot_delta_by_time(fig_dir / "mean_delta_by_time.png", prediction_rows)


def _font(size: int = 14) -> Any:
    if ImageFont is None:
        return None
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _plot_model_mse(path: Path, metrics: dict[str, Any]) -> None:
    names = list(metrics)
    vals = [metrics[n]["mse"] for n in names]
    _bar_chart(path, names, vals, "Held-out MSE by model", "MSE")


def _plot_close_ratios(path: Path, metrics: dict[str, Any]) -> None:
    names = ["ridge_time", "ridge_time_state", "time_mean"]
    labels = [f"B{B}-{name}" for B in (2, 3, 4) for name in names]
    vals = [
        metrics[str(B)]["models"][name]["close_ratio_from_means"]
        for B in (2, 3, 4)
        for name in names
    ]
    _bar_chart(path, labels, vals, "Additive close ratio from means", "ratio")


def _plot_delta_by_time(path: Path, prediction_rows: Sequence[dict[str, Any]]) -> None:
    by_t: dict[int, list[float]] = {}
    for row in prediction_rows:
        by_t.setdefault(int(row["t"]), []).append(float(row["delta"]))
    xs = sorted(by_t)
    ys = [float(np.mean(by_t[x])) for x in xs]
    w, h = 900, 520
    margin = 70
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), "Mean single-step correction value by time", fill="black", font=_font(18))
    y_min = min(ys)
    y_max = max(ys)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0
    prev = None
    for x, y in zip(xs, ys):
        px = margin + (w - 2 * margin) * x / max(xs)
        py = h - margin - (h - 2 * margin) * (y - y_min) / (y_max - y_min)
        if prev is not None:
            draw.line([prev, (px, py)], fill="#2b6cb0", width=3)
        prev = (px, py)
    draw.rectangle([margin, margin, w - margin, h - margin], outline="black")
    draw.text((margin, h - 45), "time step", fill="black", font=_font(12))
    draw.text((10, margin), "delta", fill="black", font=_font(12))
    img.save(path)


def _bar_chart(path: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    w, h = 1100, 580
    margin = 80
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), title, fill="black", font=_font(18))
    finite = [v for v in values if math.isfinite(v)]
    y_min = min(0.0, min(finite) if finite else 0.0)
    y_max = max(finite) if finite else 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0
    n = len(values)
    plot_w = w - 2 * margin
    bar_w = max(8, int(plot_w / max(n, 1) * 0.7))
    zero_y = h - margin - (h - 2 * margin) * (0.0 - y_min) / (y_max - y_min)
    draw.line([(margin, zero_y), (w - margin, zero_y)], fill="#666666", width=1)
    for i, (label, value) in enumerate(zip(labels, values)):
        cx = margin + (i + 0.5) * plot_w / n
        y = h - margin - (h - 2 * margin) * (value - y_min) / (y_max - y_min)
        color = "#2f855a" if value >= 0 else "#c53030"
        draw.rectangle([cx - bar_w / 2, min(y, zero_y), cx + bar_w / 2, max(y, zero_y)], fill=color)
        if n <= 12:
            draw.text((cx - 35, h - margin + 10), label, fill="black", font=_font(10))
    draw.rectangle([margin, margin, w - margin, h - margin], outline="black")
    draw.text((10, margin), ylabel, fill="black", font=_font(12))
    img.save(path)


def default_out_dir(prefix: Path, git_sha: str) -> Path:
    return Path(f"{prefix}_{git_sha}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--results-prefix", type=Path, default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1729)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args(argv)

    git_sha = git_short_hash()
    out_dir = args.out_dir or default_out_dir(args.results_prefix, git_sha)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already exists and is non-empty; refusing to overwrite"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_protocol_a_rows(args.input_dir)
    prediction_rows, aggregate = run_audit(
        rows,
        n_folds=args.n_folds,
        fold_seed=args.fold_seed,
        ridge_alpha=args.ridge_alpha,
    )
    config = {
        "input_dir": str(args.input_dir),
        "out_dir": str(out_dir),
        "git_sha": git_sha,
        "n_folds": args.n_folds,
        "fold_seed": args.fold_seed,
        "ridge_alpha": args.ridge_alpha,
        "state_features": list(STATE_FEATURES),
    }
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "prediction_rows.json", prediction_rows)
    write_json(out_dir / "aggregate_stats.json", aggregate)
    write_interpretation(out_dir, aggregate)
    write_figures(out_dir, aggregate, prediction_rows)
    print(json.dumps(aggregate["verdict"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
