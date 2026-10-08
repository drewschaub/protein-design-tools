# protein_design_tools/_optional.py
"""
Optional dependencies, imported in exactly one place.

PyTorch, JAX and ``requests`` are extras, not requirements.  Each name below is
``None`` when its package is not installed.  A function that needs one calls
``require_torch()`` / ``require_jax()`` / ``import_requests()`` first, so the
failure is an ImportError naming the extra to install rather than an
AttributeError on ``None`` from deep inside the computation.

Modules that annotate signatures with ``jnp.ndarray`` or ``torch.Tensor`` must
start with ``from __future__ import annotations``; otherwise the annotation is
evaluated at import time and fails when the backend is missing.
"""

from __future__ import annotations

_HINT = "Install it with: pip install 'protein-design-tools[{extra}]'"

try:
    import torch
except ImportError:
    torch = None

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
except ImportError:
    jax = None
    jnp = None

    def jit(fun):
        """Stand-in for ``jax.jit`` so ``@jit`` still works at import time."""
        return fun


def require_torch() -> None:
    """Raise an ImportError naming the ``torch`` extra if PyTorch is missing."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for this function but is not installed. "
            + _HINT.format(extra="torch")
        )


def require_jax() -> None:
    """Raise an ImportError naming the ``jax`` extra if JAX is missing."""
    if jax is None:
        raise ImportError(
            "JAX is required for this function but is not installed. "
            + _HINT.format(extra="jax")
        )


def import_requests():
    """Return the ``requests`` module, or raise an ImportError naming the extra."""
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "requests is required to download structures from RCSB but is not "
            "installed. " + _HINT.format(extra="fetch")
        ) from exc
    return requests
