"""Utilities for reusing the original YOPO implementation inside the Docker image."""

from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_YOPO_ROOT = "/workspace/YOPO/YOPO"


def add_original_yopo_to_path(root: str | None = None) -> Path:
    """Add the original YOPO Python source directory to ``sys.path``.

    The new project lives on the host-mounted ``/home/hwm`` tree, while the
    reference YOPO implementation and dataset live in ``/workspace/YOPO``.
    Keeping this bridge explicit avoids copying and silently diverging from the
    controller, lattice and differentiable cost code used by the baseline.
    """

    yopo_root = Path(root or os.environ.get("YOPO_ORIGINAL_ROOT", DEFAULT_YOPO_ROOT)).resolve()
    if not yopo_root.is_dir():
        raise FileNotFoundError(
            f"Original YOPO source was not found at {yopo_root}. "
            "Run inside the yopo_fm container or set YOPO_ORIGINAL_ROOT."
        )
    root_str = str(yopo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return yopo_root
