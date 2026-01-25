#!/usr/bin/env python3
"""
Debug helper script for HPC environment issues.

This script provides diagnostic information to help troubleshoot
environment setup problems.

Usage:
    python scripts/debug_env.py
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return output."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def check_file(path, description):
    """Check if a file/directory exists."""
    p = Path(path)
    exists = p.exists()
    symbol = "✓" if exists else "✗"
    print(f"{symbol} {description:40s} {path}")
    if exists and p.is_file():
        size = p.stat().st_size
        print(f"    Size: {size:,} bytes")
    return exists


def main():
    print("="*70)
    print("HPC Environment Debug Information")
    print("="*70)
    
    # Python info
    run_command("python --version", "Python Version")
    run_command("which python", "Python Location")
    
    # Conda info
    run_command("conda info", "Conda Info")
    
    # Environment variables
    print(f"\n{'='*70}")
    print("Key Environment Variables")
    print(f"{'='*70}\n")
    
    env_vars = [
        "CONDA_DEFAULT_ENV",
        "CUDA_HOME",
        "CUDACXX",
        "CC",
        "CXX",
        "TORCH_CUDA_ARCH_LIST",
        "TMPDIR",
        "TEMP",
        "TMP",
        "HF_HOME",
        "PIP_NO_CACHE_DIR",
        "PYTHONPATH",
        "PATH",
        "LD_LIBRARY_PATH",
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"{var:25s} = {value}")
    
    # Module info (HPC specific)
    run_command("module list 2>&1", "Loaded Modules")
    
    # Compiler info
    run_command("gcc --version | head -n 1", "GCC Version")
    run_command("which gcc", "GCC Location")
    run_command("nvcc --version | grep release", "NVCC Version")
    run_command("which nvcc", "NVCC Location")
    
    # PyTorch info
    print(f"\n{'='*70}")
    print("PyTorch Configuration")
    print(f"{'='*70}\n")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                capability = torch.cuda.get_device_capability(i)
                print(f"  Compute capability: {capability[0]}.{capability[1]}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
    except ImportError:
        print("ERROR: PyTorch not installed")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Compiled extensions
    print(f"\n{'='*70}")
    print("Compiled CUDA Extensions")
    print(f"{'='*70}\n")
    
    extensions = [
        ("flash_attn", "Flash Attention"),
        ("causal_conv1d", "Causal Conv1d"),
        ("mamba_ssm", "Mamba SSM"),
    ]
    
    for module_name, display_name in extensions:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {display_name:25s} {version}")
        except ImportError as e:
            print(f"✗ {display_name:25s} NOT INSTALLED ({e})")
    
    # Repository structure
    print(f"\n{'='*70}")
    print("Repository Structure")
    print(f"{'='*70}\n")
    
    repo_root = Path(__file__).resolve().parents[1]
    print(f"Repository root: {repo_root}\n")
    
    check_file(repo_root / "external/remdm", "ReMDM submodule")
    check_file(repo_root / "external/remdm/main.py", "ReMDM main.py")
    check_file(repo_root / "external/PRISM", "PRISM submodule")
    check_file(repo_root / "scripts/setup_hpc_env.sh", "Setup script")
    check_file(repo_root / "scripts/preflight_check.py", "Preflight check")
    check_file(repo_root / "logs/env_snapshots", "Environment snapshots dir")
    
    latest_snapshot = repo_root / "logs/env_snapshots/latest.txt"
    if check_file(latest_snapshot, "Latest environment snapshot"):
        print(f"    Created: {latest_snapshot.stat().st_mtime}")
    
    # Disk space
    print(f"\n{'='*70}")
    print("Disk Space")
    print(f"{'='*70}\n")
    
    run_command("df -h $HOME", "Home Directory")
    run_command("df -h /tmp", "Tmp Directory")
    run_command("du -sh $HOME/.cache 2>/dev/null || echo 'No cache dir'", "Cache Size")
    run_command("du -sh $HOME/.tmp 2>/dev/null || echo 'No .tmp dir'", "Tmp Dir Size")
    
    # Recent logs
    print(f"\n{'='*70}")
    print("Recent Installation Logs")
    print(f"{'='*70}\n")
    
    log_files = [
        "logs/flash_attn_install.log",
        "logs/causal_conv1d_install.log",
        "logs/mamba_ssm_install.log",
    ]
    
    for log_file in log_files:
        log_path = repo_root / log_file
        if log_path.exists():
            print(f"✓ {log_file}")
            print(f"    Last 20 lines:")
            with open(log_path) as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(f"    {line.rstrip()}")
        else:
            print(f"✗ {log_file} not found")
    
    print(f"\n{'='*70}")
    print("Debug information collection complete")
    print(f"{'='*70}\n")
    
    print("Next steps:")
    print("  - Review output above for errors")
    print("  - Check installation logs in logs/")
    print("  - Run: python scripts/preflight_check.py")
    print("  - See: docs/HPC_SETUP.md for troubleshooting")
    print()


if __name__ == "__main__":
    main()
