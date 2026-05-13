# Review Loop — Corrector Timing Statistical Framework Memo

Artifact reviewed:

- `research/corrector_timing_statistical_framework_memo.tex`
- `research/corrector_timing_statistical_framework_memo.pdf`

Date: 2026-05-11.

## Build Checks

- `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/private/tmp/corrector_timing_memo_build research/corrector_timing_statistical_framework_memo.tex` completed successfully.
- Final PDF copied to `research/corrector_timing_statistical_framework_memo.pdf`.
- `pdfinfo` reports 10 A4 pages, unencrypted, PDF 1.7.
- Log check found no unresolved citations, undefined references, fatal errors, or LaTeX errors after the final run.
- Remaining overfull/underfull box warnings are typography only.
- `git diff --check` passed.

## Mathematical Checks

- Möbius inversion statement reviewed. Initial proof wording silently included
  the empty set; fixed by explicitly setting `I(empty)=G(empty)=0` inside the
  proof.
- Ran a Python algebra check on random finite set functions verifying exact
  Möbius reconstruction.
- Ran a Python algebra check verifying that nonpositive pair coefficients imply
  diminishing returns for the second-order set function.
- Marginal and pairwise surrogate-regret propositions are standard sup-norm
  comparison arguments; constants are checked as `2 eta`, `2 zeta`, and
  `2 alpha` under the stated two-sided uniform error definitions.
- Robustness corollary `|DR_R| <= 4 rho` is valid because `DR_R` is a signed
  sum of four residual terms.
- KL contraction proposition is stated only under an explicit contraction
  assumption. The memo does not assert that ProSeCo satisfies this assumption.
- Pinsker/downstream-sensitivity paragraph gives a bound on downstream bias,
  not a signed terminal-improvement guarantee.

## Scope Checks

- The memo frames the contribution as a diagnostic/statistical framework, not a
  scheduler paper.
- It explicitly avoids claiming:
  - ProSeCo is locally balanced;
  - corrector timing is submodular in general;
  - Bayesian optimization is justified by current diagnostics;
  - ProSeCo-OWT proves model-agnostic empirical behavior;
  - search is the theoretical contribution.
- ProSeCo-OWT is presented as one empirical case study / regime diagnosis.
- Locally balanced proposals and scan Gibbs are used as principled vocabulary
  and inspiration, not as direct theorem transfers.

## Source Checks

- Informed correctors source checked against arXiv `2407.21243`: title,
  authors, hollow-transformer/corrector role, and scope align with the memo.
- Zanella locally balanced proposals source checked against arXiv `1711.07424`:
  local MCMC, locally balanced proposals, and Peskun-optimality scope align.
- Adaptive Gibbs source checked against arXiv `1801.09299`: selection
  probabilities and spectral-gap adaptation scope align.
- Gibbs entropy contraction source checked against arXiv `2410.00858`:
  random-scan Gibbs entropy contraction under log-concavity; used only as
  contraction intuition.
- Lavenant-Zanella source checked against arXiv `2510.25544`: error bounds and
  optimal predictor schedule sizes; memo distinguishes this from corrector
  timing.

## Remaining Caveats

- This is a supervisor-facing theory memo, not a final thesis chapter.
- The contraction-index section is an idealized theoretical bridge. It should
  remain clearly separated from proved ProSeCo claims unless a later kernel
  analysis verifies the contraction assumption.
- The bibliography is standalone inside the memo and should be reconciled with
  `thesis/references.bib` only if this text is moved into the thesis.
