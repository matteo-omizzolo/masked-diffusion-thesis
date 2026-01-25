# Results Directory

Job outputs are saved here with timestamp: `YYYYMMDD_HHMMSS_remdm/`

## Structure

```
results/20260125_141251_remdm/          # Example successful run
├── summary.json                         # Metrics: MAUVE, entropy, perplexity
├── samples.pt                          # Generated tokens (PyTorch tensor)
├── meta.json                           # Run configuration
└── external_remdm/
    ├── generated_sequences.json        # Full text samples
    ├── config_tree.txt                 # Complete Hydra config
    └── main.log                        # Upstream ReMDM log
```

## Recent Successful Run

**Job 9575** (Smoke Test - Success):
- Timestamp: 20260125_141251
- Config: wikitext2, 16 steps, remdm-rescale
- Duration: ~1 minute
- Output files: Cleaned during repo cleanup, but structure preserved above

## Viewing Results

```bash
# Find latest result
ls -lt results/ | head -5

# View summary
cat results/<timestamp>_remdm/summary.json

# View generated text
python -c "import json; print(json.load(open('results/<timestamp>_remdm/external_remdm/generated_sequences.json'))['text_samples'][0])"
```

## Successful Job Logs

Last successful smoke test logs preserved in:
- `out/remdm_smoke_9575.out` - Full job output
- `err/remdm_smoke_9575.err` - Stderr (warnings only)
