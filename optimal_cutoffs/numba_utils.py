"""Centralized Numba import utilities for JIT compilation support.

This module provides a single location for Numba imports and fallback logic,
ensuring consistent behavior across all modules that use JIT compilation.
"""

try:
    from numba import float64, int32, jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Define dummy decorators for when numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(*args, **kwargs):
        return range(*args, **kwargs)

    float64 = float
    int32 = int


__all__ = ["NUMBA_AVAILABLE", "jit", "prange", "float64", "int32"]
