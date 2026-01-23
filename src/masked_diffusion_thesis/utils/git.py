from __future__ import annotations

import subprocess


def get_git_commit_hash() -> str:
    """Return current git commit hash, or 'UNKNOWN' if not available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "UNKNOWN"
