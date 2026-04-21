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

__all__ = [
    "paired_bootstrap_ci",
    "spearman_bootstrap_ci",
    "cohen_d",
    "classify_tier",
    "paired_t_test",
    "mean_se",
]
