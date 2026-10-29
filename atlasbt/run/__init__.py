"""atlasbt.run -- the atlas runner: plan / run / atlas / replicate (docs/RUNNING.md).

    python -m atlasbt.run plan
    python -m atlasbt.run run --ids S050,S103
    python -m atlasbt.run atlas
    python -m atlasbt.run replicate
"""
from .fastportal import FastPortal  # noqa: F401
