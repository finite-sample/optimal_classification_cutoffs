"""Advanced algorithms for threshold optimization.

This namespace contains all the specific optimization algorithms that
were previously at the top level. These are for power users who want
explicit control over the algorithm choice.

Organized by problem type:
- algorithms.binary.*
- algorithms.multiclass.*
- algorithms.multilabel.*
- algorithms.expected.*
"""

# Make submodules available
from . import binary as binary
from . import expected as expected
from . import multiclass as multiclass
from . import multilabel as multilabel

# Also provide direct access to the most commonly used ones
from .binary import exact_f1, gradient_ascent, scipy_optimize
from .multiclass import coordinate_ascent, ovr_independent, ovr_margin
from .multilabel import macro_independent, micro_coordinate_ascent

__all__ = [
    "binary",
    "multiclass",
    "multilabel",
    "expected",
    "exact_f1",
    "scipy_optimize",
    "gradient_ascent",
    "ovr_independent",
    "ovr_margin",
    "coordinate_ascent",
    "macro_independent",
    "micro_coordinate_ascent",
]
