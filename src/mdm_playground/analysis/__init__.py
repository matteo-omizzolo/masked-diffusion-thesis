"""Phase 2 analysis package.

See docs/thesis/experiments/ANALYSIS_SPEC.md for the contract this module
implements.
"""

from .stats import (
    paired_bootstrap_ci,
    spearman_bootstrap_ci,
    cohen_d,
    classify_tier,
    paired_t_test,
    mean_se,
)
from .combinatorial_diagnostics import (
    schedule_jaccard,
    mean_pairwise_jaccard,
    random_jaccard_baseline,
    overlap_diagnostics,
    variance_decomposition,
    build_combinatorial_diagnostics,
)
from .theorem_a_constants import (
    residual_sigma_xi,
    proxy_rank_rho,
    gain_scale_sigma_delta,
    interaction_gamma_upper,
    low_gain_share,
    theorem_a_plugin_bound,
    build_theorem_a_constants,
)

__all__ = [
    "paired_bootstrap_ci",
    "spearman_bootstrap_ci",
    "cohen_d",
    "classify_tier",
    "paired_t_test",
    "mean_se",
    "schedule_jaccard",
    "mean_pairwise_jaccard",
    "random_jaccard_baseline",
    "overlap_diagnostics",
    "variance_decomposition",
    "build_combinatorial_diagnostics",
    "residual_sigma_xi",
    "proxy_rank_rho",
    "gain_scale_sigma_delta",
    "interaction_gamma_upper",
    "low_gain_share",
    "theorem_a_plugin_bound",
    "build_theorem_a_constants",
]
