"""Protocol C — bounded adaptive-controller pilot on OWT artefacts.

This module implements the CPU-only pilot specified in
``docs/thesis/next_steps/ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md`` and
``docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`` §4.2.

The pipeline reuses Phase 1 OWT Protocol A trajectories (50 seeds × T=64 with
per-step Δ_t and per-step signals), Phase 2b OWT MC oracle headroom, and
Phase 2b Refinement A′ σ_ξ to deliver:

- ε(s)        — least-squares calibration error of the linear proxy
                ψ_linear(s_t) = a · s_t + b on Phase 1 trajectories.
- ε̃(s)       — bucket-mean calibration error of ψ̃_bucket(z_t) where
                z_t = (signal_quartile, phase_bucket).
- ε̃ / ε      — ratio per signal; Q-adapt-2 quantitative answer.
- λ(s, B)    — Lagrangian multiplier tuned so 𝔼_seed[|σ_λ|] = B.
- Δ_close_A  — additive-surrogate close ratio for σ_λ and σ_topB on the
                Phase 1 trajectories, paired against the uniform schedule.
- σ_ξ · √B / √2  — Refinement-A′ uncertainty band on the additive surrogate
                vs. true G gap.
- Hamming overlap — diagnostic of σ_λ vs. best Phase-2b MC schedule per seed.

The module is pure (no I/O); see ``scripts/run_protocol_c_owt.py`` for the
artefact-loading entry script.

References
----------
- Activation audit: ``docs/thesis/next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md``
- Theory: ``docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`` §2.1, §4.2
- Experiment plan: ``docs/thesis/next_steps/ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md``
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

#: A Phase-1 trajectory is a dict with keys ``seed``, ``T``, ``per_t``.
#: Each entry of ``per_t`` is a dict with at least:
#:   ``t``, ``delta``, ``entropy``, ``inverse_margin``, ``quality_mass_proxy``,
#:   ``unmasked_fraction``, ``n_revisable``, ``n_masked``.
Trajectory = Mapping[str, Any]
PerStep = Mapping[str, Any]

#: A bucket key ``(signal_bucket, phase_bucket)``.
BucketKey = tuple[int, int]

#: A schedule is a frozenset of step indices.
Schedule = frozenset[int]


SIGNAL_KINDS: tuple[str, ...] = (
    "entropy",
    "inverse_margin",
    "quality_mass_proxy",
)


# ---------------------------------------------------------------------------
# State bucketing
# ---------------------------------------------------------------------------


def compute_signal_thresholds(
    trajectories: Sequence[Trajectory],
    signal_kind: str,
    n_signal_bins: int = 4,
) -> tuple[float, ...]:
    """Compute signal-bucket thresholds (quantiles) over all (seed, t) pairs.

    Parameters
    ----------
    trajectories : sequence of Trajectory
        Phase 1 trajectories.
    signal_kind : str
        One of ``SIGNAL_KINDS``.
    n_signal_bins : int, default 4
        Number of quantile bins.

    Returns
    -------
    thresholds : tuple of floats, length ``n_signal_bins - 1``
        Internal cut points; bucket 0 is ``s ≤ thresholds[0]``, bucket
        ``n_signal_bins - 1`` is ``s > thresholds[-1]``. Strictly increasing.
    """
    if signal_kind not in SIGNAL_KINDS:
        msg = f"signal_kind {signal_kind!r} not in {SIGNAL_KINDS}"
        raise ValueError(msg)
    if n_signal_bins < 2:
        msg = "n_signal_bins must be >= 2"
        raise ValueError(msg)

    values: list[float] = []
    for traj in trajectories:
        for step in traj["per_t"]:
            values.append(float(step[signal_kind]))
    if not values:
        msg = "no signal values found in trajectories"
        raise ValueError(msg)

    arr = np.asarray(values, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_signal_bins + 1)[1:-1]
    thresholds = np.quantile(arr, qs)
    return tuple(float(x) for x in thresholds)


def bucket_signal(
    s_value: float, thresholds: Sequence[float]
) -> int:
    """Return the signal bucket index given the precomputed quantile thresholds."""
    for i, thr in enumerate(thresholds):
        if s_value <= thr:
            return i
    return len(thresholds)


def bucket_phase(t: int, T: int, n_phase_bins: int = 3) -> int:
    """Return the phase bucket index for step ``t`` in horizon ``T``.

    With ``n_phase_bins=3`` the buckets are early=t<T/3, mid=T/3<=t<2T/3,
    late=t>=2T/3.
    """
    if T <= 0:
        msg = "T must be positive"
        raise ValueError(msg)
    if n_phase_bins < 1:
        msg = "n_phase_bins must be >= 1"
        raise ValueError(msg)
    if t < 0:
        msg = "t must be >= 0"
        raise ValueError(msg)
    edges = [int(round(T * k / n_phase_bins)) for k in range(1, n_phase_bins)]
    for i, e in enumerate(edges):
        if t < e:
            return i
    return n_phase_bins - 1


def bucket_state(
    s_value: float,
    t: int,
    T: int,
    signal_thresholds: Sequence[float],
    n_phase_bins: int = 3,
) -> BucketKey:
    """Return the bucket ``(signal_bucket, phase_bucket)`` for state z_t."""
    return (
        bucket_signal(s_value, signal_thresholds),
        bucket_phase(t, T, n_phase_bins),
    )


# ---------------------------------------------------------------------------
# Bucket-mean ψ̃ estimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketModel:
    """Bucket-mean Δ_t estimator ψ̃_bucket(z) = mean(Δ | z).

    Attributes
    ----------
    signal_kind : str
    signal_thresholds : tuple of floats
    n_phase_bins : int
    means : mapping from BucketKey to float
        Bucket-mean Δ_t over the input trajectories.
    fallback_mean : float
        Pooled mean of Δ_t over all (seed, t); used when a bucket is empty.
    bucket_counts : mapping from BucketKey to int
    """

    signal_kind: str
    signal_thresholds: tuple[float, ...]
    n_phase_bins: int
    means: Mapping[BucketKey, float]
    fallback_mean: float
    bucket_counts: Mapping[BucketKey, int]

    def psi(self, s_value: float, t: int, T: int) -> float:
        """Return ψ̃_bucket(z_t) for the given state."""
        key = bucket_state(
            s_value, t, T, self.signal_thresholds, self.n_phase_bins
        )
        return float(self.means.get(key, self.fallback_mean))


def build_bucket_model(
    trajectories: Sequence[Trajectory],
    signal_kind: str,
    n_signal_bins: int = 4,
    n_phase_bins: int = 3,
) -> BucketModel:
    """Build the BucketModel ψ̃_bucket(z) from Phase 1 OWT trajectories."""
    signal_thresholds = compute_signal_thresholds(
        trajectories, signal_kind, n_signal_bins
    )
    sums: dict[BucketKey, float] = {}
    counts: dict[BucketKey, int] = {}
    delta_sum = 0.0
    delta_n = 0
    for traj in trajectories:
        T = int(traj["T"])
        for step in traj["per_t"]:
            t = int(step["t"])
            delta = float(step["delta"])
            s_value = float(step[signal_kind])
            key = bucket_state(
                s_value, t, T, signal_thresholds, n_phase_bins
            )
            sums[key] = sums.get(key, 0.0) + delta
            counts[key] = counts.get(key, 0) + 1
            delta_sum += delta
            delta_n += 1
    if delta_n == 0:
        msg = "no per-step deltas in trajectories"
        raise ValueError(msg)
    means = {key: sums[key] / counts[key] for key in sums}
    fallback_mean = delta_sum / delta_n
    return BucketModel(
        signal_kind=signal_kind,
        signal_thresholds=signal_thresholds,
        n_phase_bins=n_phase_bins,
        means=means,
        fallback_mean=fallback_mean,
        bucket_counts=counts,
    )


# ---------------------------------------------------------------------------
# Calibration: ε vs ε̃
# ---------------------------------------------------------------------------


def compute_eps_linear(
    trajectories: Sequence[Trajectory], signal_kind: str
) -> float:
    """Least-squares ε for the linear proxy ψ_linear(s) = a·s + b.

    Returns
    -------
    eps : float
        RMS residual of the optimal linear fit Δ_t ~ a · s_t + b across all
        (seed, t).
    """
    s_list: list[float] = []
    d_list: list[float] = []
    for traj in trajectories:
        for step in traj["per_t"]:
            s_list.append(float(step[signal_kind]))
            d_list.append(float(step["delta"]))
    if not s_list:
        msg = "no points for linear calibration"
        raise ValueError(msg)
    s_arr = np.asarray(s_list, dtype=np.float64)
    d_arr = np.asarray(d_list, dtype=np.float64)
    # Least-squares fit Δ ≈ a·s + b.
    design = np.stack([s_arr, np.ones_like(s_arr)], axis=1)
    coef, *_ = np.linalg.lstsq(design, d_arr, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a * s_arr + b
    resid = d_arr - pred
    return float(math.sqrt(float(np.mean(resid * resid))))


def compute_eps_tilde(
    trajectories: Sequence[Trajectory], model: BucketModel
) -> float:
    """Bucket-mean ε̃ for ψ̃_bucket(z) on the same dataset."""
    sq_sum = 0.0
    n = 0
    for traj in trajectories:
        T = int(traj["T"])
        for step in traj["per_t"]:
            t = int(step["t"])
            s_value = float(step[model.signal_kind])
            delta = float(step["delta"])
            psi = model.psi(s_value, t, T)
            r = delta - psi
            sq_sum += r * r
            n += 1
    if n == 0:
        msg = "no points for bucket calibration"
        raise ValueError(msg)
    return float(math.sqrt(sq_sum / n))


# ---------------------------------------------------------------------------
# Threshold-λ tuning
# ---------------------------------------------------------------------------


def policy_size_at_lambda(
    trajectories: Sequence[Trajectory],
    model: BucketModel,
    lam: float,
    *,
    cap_B: int | None = None,
) -> float:
    """Mean expected schedule size 𝔼_seed[|σ_λ|] across trajectories.

    Parameters
    ----------
    cap_B : int or None
        If provided, the per-seed size is capped at ``cap_B`` (matching the
        :func:`threshold_schedule` behaviour). Used by :func:`tune_lambda` so
        that the tuned λ reflects the policy actually deployed.
    """
    sizes = []
    for traj in trajectories:
        T = int(traj["T"])
        n = 0
        for step in traj["per_t"]:
            t = int(step["t"])
            s_value = float(step[model.signal_kind])
            psi = model.psi(s_value, t, T)
            if psi > lam:
                n += 1
        if cap_B is not None and n > cap_B:
            n = cap_B
        sizes.append(n)
    return float(np.mean(sizes))


def tune_lambda(
    trajectories: Sequence[Trajectory],
    model: BucketModel,
    target_B: int,
    n_grid: int = 4096,
) -> float:
    """Tune λ so the *capped* mean schedule size equals target_B.

    The policy actually deployed (:func:`threshold_schedule`) caps each
    trajectory's schedule at B by keeping the largest ψ̃ values. With the
    cap, mean schedule size monotonically increases from 0 toward B as λ
    decreases, plateauing at exactly B once enough seeds have ≥ B
    candidates above the threshold. This function picks the *largest* λ at
    which capped mean size equals target_B (i.e., the policy is
    "selective" — ψ̃ matters, the cap is fully active).

    With piecewise-constant ψ̃ on a small number of buckets, the size
    function is a step function; this routine finds the smallest λ where
    the capped size reaches target_B.
    """
    if target_B < 1:
        msg = "target_B must be >= 1"
        raise ValueError(msg)
    psis = sorted(set(model.means.values()) | {model.fallback_mean})
    if not psis:
        msg = "empty bucket model"
        raise ValueError(msg)
    psi_min = min(psis) - 1.0
    psi_max = max(psis) + 1.0
    grid = list(np.linspace(psi_min, psi_max, n_grid))
    candidates = sorted(set(psis + grid))
    # Sort descending; we want the largest λ at which capped size >= target_B.
    # As λ decreases (we go down the sorted candidates), capped size
    # monotonically increases. The first candidate at which capped size >=
    # target_B is the answer; it tightly hits target_B because once the cap
    # is binding for most seeds, additional λ decreases don't change size.
    best = candidates[0]
    for lam in reversed(candidates):
        size = policy_size_at_lambda(
            trajectories, model, lam, cap_B=target_B
        )
        if size >= target_B - 1e-9:
            best = lam
            break
    else:
        best = candidates[0]
    return float(best)


# ---------------------------------------------------------------------------
# Schedules and additive surrogate
# ---------------------------------------------------------------------------


def threshold_schedule(
    trajectory: Trajectory, model: BucketModel, lam: float, B: int
) -> Schedule:
    """σ_λ on a single trajectory, capped at B by keeping the largest ψ̃ values."""
    T = int(trajectory["T"])
    scored: list[tuple[float, int]] = []
    for step in trajectory["per_t"]:
        t = int(step["t"])
        s_value = float(step[model.signal_kind])
        psi = model.psi(s_value, t, T)
        if psi > lam:
            scored.append((psi, t))
    if len(scored) > B:
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:B]
    return frozenset(int(t) for _, t in scored)


def topB_bucket_schedule(  # noqa: N802 — match thesis notation
    trajectory: Trajectory, model: BucketModel, B: int
) -> Schedule:
    """σ_topB on a single trajectory using ψ̃_bucket as the score."""
    T = int(trajectory["T"])
    scored: list[tuple[float, int]] = []
    for step in trajectory["per_t"]:
        t = int(step["t"])
        s_value = float(step[model.signal_kind])
        psi = model.psi(s_value, t, T)
        scored.append((psi, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return frozenset(int(t) for _, t in scored[:B])


def uniform_schedule(T: int, B: int) -> Schedule:
    """Uniform-spaced schedule of size B in {0, …, T-1}."""
    if B <= 0:
        return frozenset()
    if B >= T:
        return frozenset(range(T))
    step = T / B
    return frozenset(int(i * step) for i in range(B))


def additive_surrogate(trajectory: Trajectory, schedule: Iterable[int]) -> float:
    """A(S; trajectory) = Σ_{t ∈ S} Δ_t under the realised per-trajectory Δ_t."""
    deltas_by_t = {int(step["t"]): float(step["delta"]) for step in trajectory["per_t"]}
    return float(sum(deltas_by_t.get(int(t), 0.0) for t in schedule))


def hamming(schedule_a: Iterable[int], schedule_b: Iterable[int], T: int) -> int:
    """Hamming distance between two schedules represented as index sets."""
    sa = frozenset(int(t) for t in schedule_a)
    sb = frozenset(int(t) for t in schedule_b)
    if not (
        all(0 <= t < T for t in sa) and all(0 <= t < T for t in sb)
    ):
        msg = "schedule index out of range"
        raise ValueError(msg)
    return len(sa.symmetric_difference(sb))


# ---------------------------------------------------------------------------
# MC schedule overlap diagnostic
# ---------------------------------------------------------------------------


def best_mc_schedule(
    mc_rows_for_seed: Sequence[Mapping[str, Any]], B: int
) -> tuple[Schedule, float]:
    """Pick the highest-G MC schedule for a (seed, B). Returns (schedule, G)."""
    rows = [r for r in mc_rows_for_seed if int(r["B"]) == B]
    if not rows:
        msg = f"no MC rows for B={B}"
        raise ValueError(msg)
    best = max(rows, key=lambda r: float(r["G"]))
    return (
        frozenset(int(s) for s in best["schedule_steps"]),
        float(best["G"]),
    )


# ---------------------------------------------------------------------------
# Pipeline: protocol_c_pipeline
# ---------------------------------------------------------------------------


def protocol_c_pipeline(
    phase1_trajectories: Sequence[Trajectory],
    phase2b_mc_rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    delta_open_per_B: Mapping[int, float],
    sigma_xi_per_B: Mapping[int, float],
    *,
    B_values: Sequence[int] = (2, 3, 4),
    signal_kinds: Sequence[str] = SIGNAL_KINDS,
    n_signal_bins: int = 4,
    n_phase_bins: int = 3,
) -> dict[str, Any]:
    """Run the full Protocol C pipeline.

    Returns a dict matching the schema in
    ``ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md`` §6.
    """
    if not phase1_trajectories:
        msg = "no Phase 1 trajectories"
        raise ValueError(msg)
    T = int(phase1_trajectories[0]["T"])
    if not all(int(t["T"]) == T for t in phase1_trajectories):
        msg = "Phase 1 trajectories have inconsistent T"
        raise ValueError(msg)

    summary: dict[str, Any] = {
        "meta": {
            "protocol": "C",
            "backbone": "ProSeCo-OWT",
            "n_phase1_seeds": len(phase1_trajectories),
            "n_phase2b_seeds_for_mc_overlap": len(phase2b_mc_rows_by_seed),
            "T": T,
            "B_values": list(int(b) for b in B_values),
            "signal_kinds": list(signal_kinds),
            "n_signal_bins": n_signal_bins,
            "n_phase_bins": n_phase_bins,
        },
        "data": {
            "psi_tilde_bucket": {},
            "eps": {},
            "eps_tilde": {},
            "eps_ratio": {},
            "delta_open_per_B": {str(b): float(delta_open_per_B[b]) for b in B_values},
            "sigma_xi_per_B": {str(b): float(sigma_xi_per_B[b]) for b in B_values},
            "uncertainty_band_per_B": {
                str(b): float(sigma_xi_per_B[b]) * math.sqrt(b) / math.sqrt(2.0)
                for b in B_values
            },
            "lambda_per_signal_per_B": {},
            "delta_close_threshold_per_signal_per_B": {},
            "delta_close_topB_per_signal_per_B": {},
            "delta_close_ratio_threshold_per_signal_per_B": {},
            "delta_close_ratio_topB_per_signal_per_B": {},
            "delta_close_ratio_threshold_after_uncertainty_per_signal_per_B": {},
            "delta_close_ratio_topB_after_uncertainty_per_signal_per_B": {},
            "hamming_diagnostics_per_signal_per_B": {},
            "schedule_size_diagnostics_per_signal_per_B": {},
        },
    }

    for signal in signal_kinds:
        model = build_bucket_model(
            phase1_trajectories,
            signal_kind=signal,
            n_signal_bins=n_signal_bins,
            n_phase_bins=n_phase_bins,
        )
        eps = compute_eps_linear(phase1_trajectories, signal)
        eps_tilde = compute_eps_tilde(phase1_trajectories, model)
        eps_ratio = float(eps_tilde / eps) if eps > 0 else float("inf")

        summary["data"]["psi_tilde_bucket"][signal] = {
            f"{key[0]}_{key[1]}": float(model.means[key]) for key in model.means
        }
        summary["data"]["eps"][signal] = float(eps)
        summary["data"]["eps_tilde"][signal] = float(eps_tilde)
        summary["data"]["eps_ratio"][signal] = float(eps_ratio)
        summary["data"]["lambda_per_signal_per_B"][signal] = {}
        summary["data"]["delta_close_threshold_per_signal_per_B"][signal] = {}
        summary["data"]["delta_close_topB_per_signal_per_B"][signal] = {}
        summary["data"]["delta_close_ratio_threshold_per_signal_per_B"][signal] = {}
        summary["data"]["delta_close_ratio_topB_per_signal_per_B"][signal] = {}
        summary["data"][
            "delta_close_ratio_threshold_after_uncertainty_per_signal_per_B"
        ][signal] = {}
        summary["data"][
            "delta_close_ratio_topB_after_uncertainty_per_signal_per_B"
        ][signal] = {}
        summary["data"]["hamming_diagnostics_per_signal_per_B"][signal] = {}
        summary["data"]["schedule_size_diagnostics_per_signal_per_B"][signal] = {}

        for B in B_values:
            lam = tune_lambda(phase1_trajectories, model, target_B=B)
            S_uniform = uniform_schedule(T, B)

            threshold_schedules: list[Schedule] = []
            topB_schedules: list[Schedule] = []
            threshold_a_diffs: list[float] = []
            topB_a_diffs: list[float] = []
            threshold_sizes: list[int] = []

            for traj in phase1_trajectories:
                S_lambda = threshold_schedule(traj, model, lam, B)
                S_topB = topB_bucket_schedule(traj, model, B)
                threshold_schedules.append(S_lambda)
                topB_schedules.append(S_topB)
                threshold_sizes.append(len(S_lambda))
                a_uniform = additive_surrogate(traj, S_uniform)
                a_lambda = additive_surrogate(traj, S_lambda)
                a_topB = additive_surrogate(traj, S_topB)
                threshold_a_diffs.append(a_lambda - a_uniform)
                topB_a_diffs.append(a_topB - a_uniform)

            delta_close_threshold = float(np.mean(threshold_a_diffs))
            delta_close_topB = float(np.mean(topB_a_diffs))
            delta_open = float(delta_open_per_B[B])
            uncertainty = float(sigma_xi_per_B[B]) * math.sqrt(B) / math.sqrt(2.0)

            ratio_threshold = (
                delta_close_threshold / delta_open if delta_open > 0 else float("nan")
            )
            ratio_topB = (
                delta_close_topB / delta_open if delta_open > 0 else float("nan")
            )
            ratio_threshold_after = (
                (delta_close_threshold - uncertainty) / delta_open
                if delta_open > 0
                else float("nan")
            )
            ratio_topB_after = (
                (delta_close_topB - uncertainty) / delta_open
                if delta_open > 0
                else float("nan")
            )

            # Hamming diagnostic — only for seeds that appear in Phase 2b mc_rows.
            phase1_seeds = {int(t["seed"]) for t in phase1_trajectories}
            shared_seeds = sorted(
                phase1_seeds & set(phase2b_mc_rows_by_seed.keys())
            )
            hams: list[int] = []
            best_mc_g: list[float] = []
            n_exact = 0
            for seed in shared_seeds:
                # find the trajectory and its threshold schedule
                trajs = [
                    t for t in phase1_trajectories if int(t["seed"]) == seed
                ]
                if not trajs:
                    continue
                S_lambda = threshold_schedule(trajs[0], model, lam, B)
                mc_rows = phase2b_mc_rows_by_seed[seed]
                try:
                    best_S, best_G = best_mc_schedule(mc_rows, B)
                except ValueError:
                    continue
                ham = hamming(S_lambda, best_S, T)
                hams.append(ham)
                best_mc_g.append(best_G)
                if ham == 0:
                    n_exact += 1

            summary["data"]["lambda_per_signal_per_B"][signal][str(B)] = float(lam)
            summary["data"]["delta_close_threshold_per_signal_per_B"][signal][
                str(B)
            ] = delta_close_threshold
            summary["data"]["delta_close_topB_per_signal_per_B"][signal][
                str(B)
            ] = delta_close_topB
            summary["data"][
                "delta_close_ratio_threshold_per_signal_per_B"
            ][signal][str(B)] = float(ratio_threshold)
            summary["data"][
                "delta_close_ratio_topB_per_signal_per_B"
            ][signal][str(B)] = float(ratio_topB)
            summary["data"][
                "delta_close_ratio_threshold_after_uncertainty_per_signal_per_B"
            ][signal][str(B)] = float(ratio_threshold_after)
            summary["data"][
                "delta_close_ratio_topB_after_uncertainty_per_signal_per_B"
            ][signal][str(B)] = float(ratio_topB_after)
            summary["data"]["schedule_size_diagnostics_per_signal_per_B"][signal][
                str(B)
            ] = {
                "mean_size": float(np.mean(threshold_sizes)),
                "size_min": int(min(threshold_sizes)),
                "size_max": int(max(threshold_sizes)),
                "fraction_at_B": float(
                    np.mean([1.0 if s == B else 0.0 for s in threshold_sizes])
                ),
            }
            summary["data"]["hamming_diagnostics_per_signal_per_B"][signal][
                str(B)
            ] = {
                "n_shared_seeds": len(hams),
                "mean_hamming": float(np.mean(hams)) if hams else float("nan"),
                "max_possible_hamming": 2 * B,
                "fraction_exact_match": (n_exact / len(hams)) if hams else 0.0,
                "best_mc_g_mean": float(np.mean(best_mc_g)) if best_mc_g else 0.0,
            }

    summary["verdict"] = _classify_verdict(summary)
    return summary


# ---------------------------------------------------------------------------
# Verdict classification (pre-registered decision rule)
# ---------------------------------------------------------------------------


def _classify_verdict(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered decision rule from the experiment plan §6."""
    data = summary["data"]
    best_signal: str | None = None
    best_B: int | None = None
    best_eps_ratio = float("inf")
    best_close_ratio = -float("inf")

    for signal, eps_r in data["eps_ratio"].items():
        for B_str, ratio_after in data[
            "delta_close_ratio_threshold_after_uncertainty_per_signal_per_B"
        ][signal].items():
            score = (eps_r, -ratio_after)  # smaller-eps better, larger close-ratio better
            current_best = (best_eps_ratio, -best_close_ratio)
            if score < current_best:
                best_signal = signal
                best_B = int(B_str)
                best_eps_ratio = float(eps_r)
                best_close_ratio = float(ratio_after)

    outcome_class: str
    if best_signal is None:
        outcome_class = "inconclusive"
    elif best_eps_ratio <= 0.7 and best_close_ratio >= 0.5:
        outcome_class = "preliminary_positive"
    elif best_eps_ratio > 0.9 or best_close_ratio < 0.3:
        outcome_class = "honest_negative"
    else:
        outcome_class = "inconclusive"

    return {
        "best_signal": best_signal,
        "best_B": best_B,
        "eps_ratio_at_best": float(best_eps_ratio),
        "delta_close_ratio_at_best_after_uncertainty": float(best_close_ratio),
        "outcome_class": outcome_class,
        "decision_rule": (
            "preliminary_positive: eps_ratio <= 0.7 AND "
            "delta_close_ratio_threshold_after_uncertainty >= 0.5; "
            "honest_negative: eps_ratio > 0.9 OR "
            "delta_close_ratio_threshold_after_uncertainty < 0.3; "
            "inconclusive otherwise."
        ),
    }
