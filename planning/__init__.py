"""Planning utilities package.

This module uses lazy attribute imports to avoid import-time cycles between
packages that may depend on each other.
"""

from __future__ import annotations

__all__ = ["MPPI", "SimNorm", "SAM", "InfoProp"]


def __getattr__(name: str):
    if name == "MPPI":
        from planning.mppi import MPPI

        return MPPI
    if name == "SimNorm":
        from planning.sim_norm import SimNorm

        return SimNorm
    if name == "SAM":
        from planning.sam_optimizer import SAM

        return SAM
    if name == "InfoProp":
        from planning.info_prop import InfoProp

        return InfoProp
    raise AttributeError(f"module 'planning' has no attribute {name!r}")
