"""Ghi lại phiên bản thư viện/phần cứng để tái lập kết quả."""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata

from vehicledet.paths import REPO_ROOT

_PACKAGES = ["torch", "torchvision", "ultralytics", "numpy", "opencv-python", "PyYAML"]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def env_info() -> dict:
    info: dict = {"python": platform.python_version(), "platform": platform.platform()}
    info["packages"] = {}
    for pkg in _PACKAGES:
        try:
            info["packages"][pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info["packages"][pkg] = None
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["cuda"], info["gpu"] = None, None
    info["git_commit"] = _git("rev-parse", "HEAD")
    info["git_dirty"] = bool(_git("status", "--porcelain"))
    return info
