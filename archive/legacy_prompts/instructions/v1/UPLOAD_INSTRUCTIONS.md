# How to use this package with Claude Code

## Files included

1. `MSc_thesis_direction_brief.md`
   - research summary and refocused thesis direction.

2. `claude_code_repo_update_prompt.md`
   - prompt asking Claude Code to clean the repo, update docs, archive/remove obsolete files, and bootstrap the experimental setup.

3. `claude_code_math_analysis_prompt.md`
   - prompt asking Claude Code to start the mathematical analysis and derivation work in a traceable way.

## Recommended usage order

### First pass
Upload:
- the repository,
- `MSc_thesis_direction_brief.md`,
- `claude_code_repo_update_prompt.md`.

Ask Claude Code to do the repo refocus and documentation cleanup first.

### Second pass
Once the repo is cleaner, upload or reference:
- `claude_code_math_analysis_prompt.md`.

Ask Claude Code to start the proof worklog and theorem exploration.

## Placeholders you may want to add manually before uploading

If you know them, it may help to add or tell Claude:

- the repo root or branch you want edited,
- which papers you already read,
- how aggressive you want file deletion to be,
- whether it may clone external repos immediately,
- any compute constraints,
- whether you want archived files moved or kept in place with deprecation banners.

## Suggested short instruction to accompany the upload

Use these files as the main context. First clean and refocus the repository around the realistic thesis direction in the brief. Then update the reading plan based on what has already been read, archive or remove obsolete material conservatively, and bootstrap the experimental infrastructure around MDLM / ReMDM / ProSeCo first. After that, start the mathematical worklog using the separate math-analysis prompt.

