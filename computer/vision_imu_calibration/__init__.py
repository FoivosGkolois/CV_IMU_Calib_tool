"""PC-first Vision–IMU calibration feasibility prototype."""

from .geometry import KneeAngleResult, calculate_knee_flexion_deg
from .models import Landmark, LegLandmarks, LegSide, PoseFrame

__all__ = [
    "KneeAngleResult",
    "Landmark",
    "LegLandmarks",
    "LegSide",
    "PoseFrame",
    "calculate_knee_flexion_deg",
]
