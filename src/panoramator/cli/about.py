from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import sys
import tomllib

import cv2

from panoramator.config.models import PanoramaConfig

ROOT_DIR = Path(__file__).resolve().parents[3]
ASCII_LOGO_PATH = ROOT_DIR / "assets" / "ascii.txt"
SUPPORTED_OUTPUT_FORMATS = ("PNG", "JPEG", "WebP", "TIFF")


def read_ascii_logo() -> str:
    if ASCII_LOGO_PATH.exists():
        return ASCII_LOGO_PATH.read_text(encoding="utf-8").rstrip("\n")
    return "Panoramator"


def format_label(value: str) -> str:
    return value.replace("_", " ").title()


def format_opencv_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return version


def detect_backend() -> str:
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return "CUDA"
    except (AttributeError, cv2.error):
        pass
    return "CPU"


def get_package_version() -> str:
    if is_running_from_source_tree():
        pyproject_version = get_pyproject_version()
        if pyproject_version != "0.0.0":
            return pyproject_version
    try:
        return importlib.metadata.version("panoramator")
    except importlib.metadata.PackageNotFoundError:
        return get_pyproject_version()


def is_running_from_source_tree() -> bool:
    return Path(__file__).resolve().parents[2] == ROOT_DIR / "src"


def get_pyproject_version() -> str:
    pyproject_path = ROOT_DIR / "pyproject.toml"
    if not pyproject_path.exists():
        return "0.0.0"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = project.get("version")
    if isinstance(version, str) and version:
        return version
    return "0.0.0"


def compose_columns(left_lines: list[str], right_lines: list[str], gap: int = 4) -> str:
    left_width = max((len(line) for line in left_lines), default=0)
    rows: list[str] = []
    total_rows = max(len(left_lines), len(right_lines))
    for index in range(total_rows):
        left = left_lines[index] if index < len(left_lines) else ""
        right = right_lines[index] if index < len(right_lines) else ""
        if right:
            rows.append(f"{left.ljust(left_width)}{' ' * gap}{right}")
        else:
            rows.append(left)
    return "\n".join(rows)


def format_version_screen() -> str:
    config = PanoramaConfig()
    version = get_package_version()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    opencv_version = format_opencv_version(cv2.__version__)
    threads = cv2.getNumThreads() or os.cpu_count() or 1
    right_lines = [
        f"Panoramator v{version}",
        "------------------------------",
        f"Python      {python_version}",
        f"OpenCV      {opencv_version}",
        f"Backend     {detect_backend()}",
        f"Threads     {threads}",
        f"Projection  {format_label(config.projection)}",
        f"Output      {'/'.join(SUPPORTED_OUTPUT_FORMATS)}",
    ]
    return compose_columns(read_ascii_logo().splitlines(), right_lines)
