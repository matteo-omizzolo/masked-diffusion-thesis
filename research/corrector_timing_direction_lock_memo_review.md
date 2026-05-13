# Codex Review Loop -- Research Direction Memo and Direction Lock

Date: 2026-05-11

Reviewed files:
- `research/corrector_timing_direction_lock_memo.tex`
- `research/corrector_timing_direction_lock_memo.pdf`

## Verdict

PASS.

The memo and active direction docs now have the requested form:
- no first-person thesis pitch;
- no alternative-framing section;
- statistical setup, theorem candidates, and empirical plan retained;
- removed the previous "what to tell the supervisor" and "what not to claim"
  sections;
- added a direct accounting of which questions are already answered by the
  current ProSeCo-OWT evidence;
- added a scope section explaining what "model-agnostic" means and what it
  does not mean;
- added recommended next steps based on the evidence;
- updated the PDF after the held-out state-predictability audit completed.

## Substantive review

The current evidence already answers much of the proposed thesis question on
ProSeCo-OWT:

- correction timing is non-vacuous;
- simple uncertainty/ranker signals do not recover the headroom;
- simple adaptive state conditioning is negative for current features;
- true-G search finds useful timing structure;
- interactions are structured and mostly redundant;
- BO/submodular theory is not currently supported strongly enough.

Therefore the next step should not be a new online scheduler from the current
feature set. The held-out state-predictability audit has now been run and is
negative: time+state prediction does not improve over time-only prediction on
held-out seeds. The evidence supports writing and synthesis first, with an
external-validity gate only if the supervisor requires it.

## Mathematical checks

- State-predictability result is a standard nested conditional-expectation /
  \(L^2\)-projection identity.
- Finite-pool regret certificate has the correct constant `2 eta + 2 alpha`.
- Quadratic interaction lemma has the correct sign:
  `DR gap = - sum_{t in B \\ A} b_{x,t}`.
- The memo separates model-agnostic definitions from ProSeCo-OWT empirical
  claims.

## Model-scope review

The memo correctly states:

- The framework is model-agnostic only as a diagnostic formalism over a fixed
  `(predictor, corrector, F, B)` tuple.
- The empirical regime classification is not model-agnostic; current evidence
  is ProSeCo-OWT-specific.
- Heavy LLaDA/ProSeCo-LLaDA experiments should be external-validity gates only,
  because the current bounded LLaDA-SFT probe did not show clear positive
  oracle headroom.

## Commands run

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error corrector_timing_direction_lock_memo.tex
latexmk -c corrector_timing_direction_lock_memo.tex
pdfinfo research/corrector_timing_direction_lock_memo.pdf
pdftotext research/corrector_timing_direction_lock_memo.pdf -
python3.11 -m py_compile scripts/proseco/state_predictability/analyze_state_predictability.py scripts/proseco/landscape/analyze_set_function_structure.py scripts/proseco/landscape/analyze_schedule_landscape_geometry.py scripts/proseco/landscape/analyze_schedule_neighborhood_diagnostics.py scripts/proseco/landscape/run_schedule_neighborhood_diagnostics.py scripts/proseco/interactions/validate_phase1_schedule_level_b2.py scripts/proseco/interactions/validate_phase1_schedule_level_b34.py
pytest tests/proseco/state_predictability/test_state_predictability.py tests/proseco/interactions/test_phase1_schedule_validation_b2.py tests/proseco/interactions/test_phase1_schedule_validation_b34.py tests/proseco/landscape/test_set_function_structure.py tests/proseco/landscape/test_schedule_landscape_geometry.py tests/proseco/landscape/test_schedule_neighborhood_diagnostics.py -q
python3.11 -c "import mdm_playground"
python3.11 -c "import json, pathlib; base=pathlib.Path('results/state_predictability_0c39079'); json.load(open(base/'aggregate_stats.json')); json.load(open(base/'config.json')); json.load(open(base/'prediction_rows.json'))"
git diff --check
```

## Remaining caveat

The memo is supervisor-facing and intentionally compact. It should not replace
the full thesis chapters. Its purpose is to lock the research question and
decide whether the supervisor accepts ProSeCo-OWT as a case study plus a
model-agnostic diagnostic framework, or requires one additional
external-validity gate.
