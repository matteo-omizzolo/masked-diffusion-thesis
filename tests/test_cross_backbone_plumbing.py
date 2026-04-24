from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mdm_playground.scheduling.backends.checkpoint_utils import (
    detect_proseco_snapshot_backend,
)


def test_detect_proseco_snapshot_backend(tmp_path: Path) -> None:
    owt = tmp_path / "owt"
    owt.mkdir()
    (owt / "config.json").write_text("{}")
    (owt / "configuration_proseco.py").write_text("# stub")
    (owt / "modeling_proseco.py").write_text("# stub")
    assert detect_proseco_snapshot_backend(str(owt)) == "proseco_owt"

    llada = tmp_path / "llada"
    llada.mkdir()
    (llada / "config.json").write_text("{}")
    (llada / "configuration_llada.py").write_text("# stub")
    (llada / "modeling_llada.py").write_text("# stub")
    assert detect_proseco_snapshot_backend(str(llada)) == "proseco_llada_sft"


def test_protocol_a_script_surrogate_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "protocol_a"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_protocol_a_proseco_snapshot.py"
    cmd = [
        sys.executable,
        str(script),
        "--surrogate",
        "--T",
        "4",
        "--K",
        "2",
        "--seed_start",
        "10",
        "--out_dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    traj_files = sorted(out_dir.glob("trajectory_*.json"))
    assert len(traj_files) == 2
    sample = json.loads(traj_files[0].read_text())
    assert sample["T"] == 4
    assert len(sample["per_t"]) == 4
    assert {"t", "delta", "entropy", "inverse_margin", "quality_mass_proxy"} <= set(
        sample["per_t"][0].keys()
    )

