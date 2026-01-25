"""
Dtype compatibility patch for ReMDM confidence strategy.

Fixes BFloat16/Float32 mismatch in remdm-conf strategy without modifying
the upstream codebase in external/remdm.

Root cause: confident_score tensor is created as bfloat16 (line 863), 
but conf_values from p_x0 indexing is float32 (line 751), causing 
RuntimeError during index assignment (line 752).

Usage:
    # Not currently integrated - use working strategies instead
    # To enable: modify remdm_adapter.py to apply patch before sampling
    
Recommended approach: Use remdm-rescale, remdm-cap, or remdm-loop strategies
instead of patching upstream code.

Manual fix for upstream (external/remdm/diffusion.py line 752):
    # Change: conf[unmask_mask] = conf_values[unmask_mask]
    # To:     conf[unmask_mask] = conf_values[unmask_mask].to(conf.dtype)
"""

import torch
import functools
from typing import Optional


def patch_ddpm_caching_update(original_method):
    """
    Wrapper to fix dtype mismatch in _ddpm_caching_update for remdm-conf.
    
    The issue occurs at line 752 in external/remdm/diffusion.py:
        conf[unmask_mask] = conf_values[unmask_mask]
    
    Where conf is bfloat16 but conf_values is float32.
    """
    @functools.wraps(original_method)
    def wrapper(self, x, t, dt, p_x0=None, conf=None):
        # Call original method
        result = original_method(self, x, t, dt, p_x0=p_x0, conf=conf)
        
        # Result is (p_x0_cache, xs, confident_score)
        # If conf strategy is used and confident_score dtype mismatches, fix it
        if len(result) == 3 and result[2] is not None:
            p_x0_cache, xs, confident_score = result
            
            # Ensure consistent dtype (use float32 for better precision)
            if confident_score.dtype == torch.bfloat16:
                confident_score = confident_score.to(torch.float32)
            
            return p_x0_cache, xs, confident_score
        
        return result
    
    return wrapper


def apply_dtype_patch(diffusion_model):
    """
    Apply dtype patch to a ReMDM diffusion model instance.
    
    Args:
        diffusion_model: Instance from external.remdm.diffusion module
        
    Returns:
        Patched model (modifies in-place)
    """
    # Only patch if method exists
    if hasattr(diffusion_model, '_ddpm_caching_update'):
        original_method = diffusion_model._ddpm_caching_update
        diffusion_model._ddpm_caching_update = patch_ddpm_caching_update(
            original_method
        ).__get__(diffusion_model, type(diffusion_model))
    
    return diffusion_model


def create_monkey_patch_module():
    """
    Alternative approach: Monkey-patch the diffusion module before import.
    
    WARNING: This modifies the upstream code in memory. Use with caution.
    Only recommended for testing purposes.
    """
    import sys
    import importlib
    
    # Import the upstream module
    if 'external.remdm.diffusion' in sys.modules:
        diffusion_module = sys.modules['external.remdm.diffusion']
    else:
        # Add to path if needed
        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        remdm_path = os.path.join(repo_root, 'external', 'remdm')
        if remdm_path not in sys.path:
            sys.path.insert(0, remdm_path)
        
        diffusion_module = importlib.import_module('diffusion')
    
    # Patch the class method
    if hasattr(diffusion_module, 'DiscreteDiffusionMatryoshka'):
        original_class = diffusion_module.DiscreteDiffusionMatryoshka
        original_init = original_class.__init__
        
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Apply patch after initialization
            if hasattr(self, '_ddpm_caching_update'):
                self._ddpm_caching_update = patch_ddpm_caching_update(
                    self._ddpm_caching_update
                ).__get__(self, type(self))
        
        original_class.__init__ = patched_init


# Alternative: Direct source code patch (most invasive, not recommended)
DTYPE_FIX_PATCH = """
# At line 752 in external/remdm/diffusion.py, change:
# FROM:
conf[unmask_mask] = conf_values[unmask_mask]

# TO:
conf[unmask_mask] = conf_values[unmask_mask].to(conf.dtype)
"""
