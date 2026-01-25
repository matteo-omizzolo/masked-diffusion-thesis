#!/usr/bin/env python3
"""
Preflight Check for Masked Diffusion Thesis

This script verifies that all dependencies are properly installed and
the environment is ready to run experiments.

Usage:
    python scripts/preflight_check.py [--skip-gpu] [--require-mamba]

Exit codes:
    0: All checks passed
    1: Critical dependency missing
    2: GPU not available (unless --skip-gpu)
    3: Upstream ReMDM cannot be imported
"""

import argparse
import os
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_check(name: str, passed: bool, detail: str = ""):
    """Print a check result."""
    symbol = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
    status = f"{Colors.GREEN}OK{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
    detail_str = f" ({detail})" if detail else ""
    print(f"{symbol} {name:40s} [{status}]{detail_str}")


def print_warning(name: str, detail: str = ""):
    """Print a warning."""
    symbol = f"{Colors.YELLOW}⚠{Colors.RESET}"
    status = f"{Colors.YELLOW}WARN{Colors.RESET}"
    detail_str = f" ({detail})" if detail else ""
    print(f"{symbol} {name:40s} [{status}]{detail_str}")


def check_core_imports() -> bool:
    """Check core Python dependencies."""
    print_header("Core Dependencies")
    
    all_passed = True
    
    # Core packages
    required_packages = [
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("tqdm", "tqdm"),
        ("rich", "rich"),
    ]
    
    for module_name, display_name in required_packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print_check(display_name, True, version)
        except ImportError as e:
            print_check(display_name, False, str(e))
            all_passed = False
    
    return all_passed


def check_ml_frameworks() -> bool:
    """Check ML framework dependencies."""
    print_header("ML Frameworks")
    
    all_passed = True
    
    ml_packages = [
        ("lightning", "PyTorch Lightning"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("timm", "timm"),
        ("wandb", "Weights & Biases"),
    ]
    
    for module_name, display_name in ml_packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print_check(display_name, True, version)
        except ImportError as e:
            print_check(display_name, False, str(e))
            all_passed = False
    
    return all_passed


def check_compiled_extensions(require_mamba: bool = False) -> bool:
    """Check compiled CUDA extensions."""
    print_header("Compiled CUDA Extensions")
    
    all_passed = True
    
    # flash-attn (required for DiT)
    try:
        import flash_attn
        version = getattr(flash_attn, "__version__", "unknown")
        print_check("flash-attn", True, version)
    except ImportError as e:
        print_check("flash-attn", False, str(e))
        all_passed = False
        print(f"  {Colors.RED}CRITICAL: flash-attn is required for DiT backbone{Colors.RESET}")
    
    # causal-conv1d (required for Mamba/DiMamba)
    try:
        import causal_conv1d
        print_check("causal-conv1d", True, "installed")
    except ImportError as e:
        if require_mamba:
            print_check("causal-conv1d", False, str(e))
            all_passed = False
            print(f"  {Colors.RED}CRITICAL: causal-conv1d is required when --require-mamba is set{Colors.RESET}")
        else:
            print_warning("causal-conv1d", "not installed (DiMamba will not work)")
    
    # mamba-ssm (optional for Mamba/DiMamba)
    try:
        import mamba_ssm
        print_check("mamba-ssm", True, "installed")
    except ImportError as e:
        if require_mamba:
            print_check("mamba-ssm", False, str(e))
            all_passed = False
            print(f"  {Colors.RED}CRITICAL: mamba-ssm is required when --require-mamba is set{Colors.RESET}")
        else:
            print_warning("mamba-ssm", "not installed (DiMamba will not work)")
    
    return all_passed


def check_gpu() -> bool:
    """Check GPU availability and CUDA."""
    print_header("GPU & CUDA")
    
    all_passed = True
    
    try:
        import torch
        
        # CUDA availability
        cuda_available = torch.cuda.is_available()
        print_check("CUDA available", cuda_available)
        
        if not cuda_available:
            print(f"  {Colors.RED}GPU not available. Check CUDA installation and drivers.{Colors.RESET}")
            return False
        
        # CUDA version
        cuda_version = torch.version.cuda
        print_check("CUDA version", True, cuda_version)
        
        # Device count
        device_count = torch.cuda.device_count()
        print_check("GPU count", device_count > 0, str(device_count))
        
        if device_count > 0:
            # Device name
            device_name = torch.cuda.get_device_name(0)
            print_check("GPU 0", True, device_name)
            
            # Compute capability
            capability = torch.cuda.get_device_capability(0)
            compute_cap = f"{capability[0]}.{capability[1]}"
            print_check("Compute capability", True, compute_cap)
            
            # Memory
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_check("GPU memory", True, f"{mem_gb:.1f} GB")
    
    except Exception as e:
        print_check("GPU check", False, str(e))
        all_passed = False
    
    return all_passed


def check_repo_structure() -> bool:
    """Check repository structure and submodules."""
    print_header("Repository Structure")
    
    all_passed = True
    
    repo_root = Path(__file__).resolve().parents[1]
    
    # Check key directories
    key_dirs = [
        ("external/remdm", "ReMDM submodule"),
        ("external/PRISM", "PRISM submodule"),
        ("configs", "Config directory"),
        ("scripts", "Scripts directory"),
        ("src/masked_diffusion_thesis", "Source package"),
    ]
    
    for dir_path, description in key_dirs:
        full_path = repo_root / dir_path
        exists = full_path.exists()
        print_check(description, exists, str(dir_path))
        if not exists:
            all_passed = False
    
    return all_passed


def check_upstream_remdm_imports() -> bool:
    """
    Check that upstream ReMDM can be imported without errors.
    
    This is the critical test: can we import ReMDM's modules that
    depend on optional backbones (flash_attn, causal_conv1d)?
    """
    print_header("Upstream ReMDM Imports")
    
    all_passed = True
    
    repo_root = Path(__file__).resolve().parents[1]
    remdm_path = repo_root / "external" / "remdm"
    
    if not remdm_path.exists():
        print_check("ReMDM path exists", False, str(remdm_path))
        return False
    
    print_check("ReMDM path exists", True, str(remdm_path))
    
    # Add ReMDM to path
    sys.path.insert(0, str(remdm_path))
    
    # Test imports that should work without optional deps
    try:
        import utils as remdm_utils
        print_check("ReMDM utils", True)
    except ImportError as e:
        print_check("ReMDM utils", False, str(e))
        all_passed = False
    
    try:
        import dataloader as remdm_dataloader
        print_check("ReMDM dataloader", True)
    except ImportError as e:
        print_check("ReMDM dataloader", False, str(e))
        all_passed = False
    
    try:
        import diffusion as remdm_diffusion
        print_check("ReMDM diffusion", True)
    except ImportError as e:
        print_check("ReMDM diffusion", False, str(e))
        all_passed = False
    
    # Test model imports (these may fail if optional deps missing)
    # DiT requires flash_attn
    try:
        from models import dit as remdm_dit
        print_check("ReMDM DiT model", True, "flash_attn available")
    except ImportError as e:
        print_check("ReMDM DiT model", False, str(e))
        all_passed = False
        print(f"  {Colors.RED}CRITICAL: DiT model cannot be imported{Colors.RESET}")
    
    # DiMamba requires causal_conv1d
    try:
        from models import dimamba as remdm_dimamba
        print_check("ReMDM DiMamba model", True, "causal_conv1d available")
    except ImportError as e:
        print_warning("ReMDM DiMamba model", "optional - will work if using DiT only")
    
    # Test main module (but don't run it - Hydra will complain about argv)
    try:
        import main as remdm_main
        print_check("ReMDM main module", True)
    except ImportError as e:
        print_check("ReMDM main module", False, str(e))
        all_passed = False
    
    # Remove from path
    sys.path.pop(0)
    
    return all_passed


def check_environment_variables() -> bool:
    """Check important environment variables."""
    print_header("Environment Variables")
    
    all_passed = True
    
    # HF_HOME
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        print_check("HF_HOME", True, hf_home)
    else:
        print_warning("HF_HOME", "not set (will use default ~/.cache/huggingface)")
    
    # CUDA_HOME (useful during builds)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        print_check("CUDA_HOME", True, cuda_home)
    else:
        print_warning("CUDA_HOME", "not set (needed only for building extensions)")
    
    # TMPDIR
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        print_check("TMPDIR", True, tmpdir)
    else:
        print_warning("TMPDIR", "not set (may cause issues with some pip installs)")
    
    return all_passed


def main():
    """Run all preflight checks."""
    parser = argparse.ArgumentParser(description="Preflight check for masked diffusion thesis")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU checks (for CPU-only testing)")
    parser.add_argument("--require-mamba", action="store_true", help="Require Mamba/DiMamba dependencies")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}Preflight Check for Masked Diffusion Thesis{Colors.RESET}")
    print(f"{'='*70}\n")
    
    # Track results
    checks = []
    
    # Run checks
    checks.append(("Core dependencies", check_core_imports()))
    checks.append(("ML frameworks", check_ml_frameworks()))
    checks.append(("Compiled extensions", check_compiled_extensions(args.require_mamba)))
    
    if not args.skip_gpu:
        checks.append(("GPU & CUDA", check_gpu()))
    else:
        print_warning("GPU check", "skipped (--skip-gpu)")
    
    checks.append(("Repository structure", check_repo_structure()))
    checks.append(("Upstream ReMDM imports", check_upstream_remdm_imports()))
    checks.append(("Environment variables", check_environment_variables()))
    
    # Summary
    print_header("Summary")
    
    passed_count = sum(1 for _, passed in checks if passed)
    total_count = len(checks)
    
    for name, passed in checks:
        symbol = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
        print(f"{symbol} {name}")
    
    print(f"\n{Colors.BOLD}Result: {passed_count}/{total_count} checks passed{Colors.RESET}\n")
    
    # Exit code
    if passed_count == total_count:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed! Environment is ready.{Colors.RESET}\n")
        return 0
    else:
        failed = [name for name, passed in checks if not passed]
        print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed:{Colors.RESET}")
        for name in failed:
            print(f"  - {name}")
        print()
        print(f"{Colors.YELLOW}Hints:{Colors.RESET}")
        print(f"  - Run setup script: bash scripts/setup_hpc_env.sh")
        print(f"  - Check installation logs in logs/")
        print(f"  - Verify you're on a GPU node: srun --pty bash")
        print()
        
        # Determine exit code
        if not args.skip_gpu and not checks[3][1]:  # GPU check failed
            return 2
        elif not checks[5][1]:  # Upstream ReMDM imports failed
            return 3
        else:
            return 1


if __name__ == "__main__":
    sys.exit(main())
