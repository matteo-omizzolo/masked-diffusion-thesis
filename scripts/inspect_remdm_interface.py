#!/usr/bin/env python3
"""
Inspect ReMDM interface: list available scripts, configs, and example commands.

This script does NOT import upstream Python modules (safe for macOS).
It only reads files from the external/remdm submodule.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def get_repo_root() -> Path:
    """Get repository root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def get_remdm_path() -> Path:
    """Get path to external/remdm submodule."""
    return get_repo_root() / "external" / "remdm"


def list_scripts(remdm_path: Path) -> None:
    """List all shell scripts in external/remdm/scripts/."""
    print("\n" + "=" * 80)
    print("Available Scripts in external/remdm/scripts/")
    print("=" * 80)
    
    scripts_dir = remdm_path / "scripts"
    if not scripts_dir.exists():
        print(f"❌ Scripts directory not found: {scripts_dir}")
        return
    
    scripts = sorted(scripts_dir.glob("*.sh"))
    if not scripts:
        print("No .sh scripts found.")
        return
    
    for i, script in enumerate(scripts, 1):
        print(f"{i:2d}. {script.name}")
        
        # Try to extract key parameters (sampler) from script
        try:
            with open(script, "r") as f:
                content = f.read()
                # Look for sampling.sampler= in the python command
                if "sampling.sampler=" in content:
                    for line in content.split("\n"):
                        if "sampling.sampler=" in line:
                            sampler = line.split("sampling.sampler=")[-1].split()[0].strip('"').strip("'")
                            print(f"    → Sampler: {sampler}")
                            break
        except Exception:
            pass


def list_configs(remdm_path: Path) -> None:
    """List YAML configs in external/remdm/configs/."""
    print("\n" + "=" * 80)
    print("Config Structure in external/remdm/configs/")
    print("=" * 80)
    
    configs_dir = remdm_path / "configs"
    if not configs_dir.exists():
        print(f"❌ Configs directory not found: {configs_dir}")
        return
    
    # Main config
    main_config = configs_dir / "config.yaml"
    if main_config.exists():
        print(f"📄 Main config: config.yaml")
    
    # List subdirectories with configs
    for subdir in sorted(configs_dir.iterdir()):
        if subdir.is_dir():
            yaml_files = sorted(subdir.glob("*.yaml"))
            if yaml_files:
                print(f"\n📁 {subdir.name}/")
                for yaml_file in yaml_files:
                    print(f"   - {yaml_file.name}")


def show_example_commands(remdm_path: Path) -> None:
    """Show example Hydra command lines based on scripts."""
    print("\n" + "=" * 80)
    print("Example Hydra Command Lines")
    print("=" * 80)
    
    print("\n1. Training example:")
    print("   python -m main mode=train data=openwebtext model=small \\")
    print("     backbone=dit parameterization=subs \\")
    print("     loader.global_batch_size=512 \\")
    print("     sampling.steps=128 \\")
    print("     hydra.run.dir=outputs/my_run")
    
    print("\n2. Sampling example (ReMDM-conf):")
    print("   python -m main mode=sample_eval \\")
    print("     data=openwebtext-split model=small \\")
    print("     backbone=dit parameterization=subs \\")
    print("     eval.checkpoint_path=/path/to/model.ckpt \\")
    print("     sampling.sampler=remdm-conf \\")
    print("     sampling.steps=1024 \\")
    print("     sampling.nucleus_p=0.9 \\")
    print("     sampling.num_sample_batches=5000 \\")
    print("     sampling.generated_seqs_path=outputs/samples.json \\")
    print("     loader.batch_size=1 \\")
    print("     T=0 time_conditioning=false \\")
    print("     hydra.run.dir=outputs/remdm-conf")
    
    print("\n3. Sampling example (ReMDM-loop):")
    print("   python -m main mode=sample_eval \\")
    print("     sampling.sampler=remdm-loop \\")
    print("     [... other params same as above ...]")
    
    print("\n4. Perplexity evaluation:")
    print("   python -m main mode=ppl_eval \\")
    print("     data=openwebtext-split model=small \\")
    print("     eval.checkpoint_path=/path/to/model.ckpt \\")
    print("     loader.eval_batch_size=32")


def inspect_main_config(remdm_path: Path) -> None:
    """Show key sections from main config.yaml."""
    print("\n" + "=" * 80)
    print("Key Parameters in config.yaml")
    print("=" * 80)
    
    config_file = remdm_path / "configs" / "config.yaml"
    if not config_file.exists():
        print(f"❌ Main config not found: {config_file}")
        return
    
    try:
        with open(config_file, "r") as f:
            lines = f.readlines()
        
        # Extract key sections
        print("\nMode options:")
        print("  mode: train / ppl_eval / sample_eval")
        
        print("\nDiffusion types:")
        for line in lines:
            if line.strip().startswith("diffusion:"):
                print(f"  {line.strip()}")
        
        print("\nBackbone options:")
        for line in lines:
            if line.strip().startswith("backbone:"):
                print(f"  {line.strip()}")
        
        print("\nParameterization options:")
        for line in lines:
            if line.strip().startswith("parameterization:"):
                print(f"  {line.strip()}")
        
        print("\nSampling parameters:")
        in_sampling = False
        for line in lines:
            if line.startswith("sampling:"):
                in_sampling = True
            elif in_sampling and line.startswith(" ") and ":" in line:
                print(f"  {line.strip()}")
            elif in_sampling and not line.startswith(" "):
                break
    
    except Exception as e:
        print(f"Error reading config: {e}")


def inspect_one_script(remdm_path: Path, script_name: str) -> None:
    """Show detailed info about one script."""
    print("\n" + "=" * 80)
    print(f"Inspecting script: {script_name}")
    print("=" * 80)
    
    script_path = remdm_path / "scripts" / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return
    
    try:
        with open(script_path, "r") as f:
            content = f.read()
        
        # Find the srun python command
        for line in content.split("\n"):
            if "python" in line and "main" in line:
                # Clean up line continuations
                print("\nPython command:")
                print(line.strip())
    
    except Exception as e:
        print(f"Error reading script: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ReMDM interface (scripts, configs, examples)"
    )
    parser.add_argument(
        "--script", 
        type=str, 
        help="Inspect specific script (e.g., remdm-conf.sh)"
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Skip example command lines"
    )
    args = parser.parse_args()
    
    repo_root = get_repo_root()
    remdm_path = get_remdm_path()
    
    print("=" * 80)
    print("ReMDM Interface Inspector")
    print("=" * 80)
    print(f"Repo root: {repo_root}")
    print(f"ReMDM path: {remdm_path}")
    
    if not remdm_path.exists():
        print(f"\n❌ ERROR: ReMDM submodule not found at {remdm_path}")
        print("Run: git submodule update --init --recursive")
        return 1
    
    if args.script:
        inspect_one_script(remdm_path, args.script)
    else:
        list_scripts(remdm_path)
        list_configs(remdm_path)
        inspect_main_config(remdm_path)
        
        if not args.no_examples:
            show_example_commands(remdm_path)
    
    print("\n" + "=" * 80)
    print("Inspection complete!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
