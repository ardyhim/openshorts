"""Build the distributable ZIP used by the Colab notebook.

Usage (from the repo root):

    python colab/build_zip.py

The archive contains a top-level ``openshorts-free/`` folder so the
notebook can locate the project root with ``looks_like_project()`` no
matter how the ZIP was uploaded.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "openshorts-free-colab.zip"
PREFIX = "openshorts-free"

EXCLUDE_DIRS = {
    ".git",
    ".github",  # CI configs are not needed inside Colab
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "output",
    "uploads",
    "venv",
    ".venv",
}

EXCLUDE_FILES = {
    ".env",
    "churchil_queen_vertical.gif",
    "churchil_queen_vertical_short.gif",
    "demo-openshorts.mp4",
    "screenshots/split-before-after.gif",
    "colab/openshorts-free-colab.zip",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".sqlite",
    ".sqlite3",
    ".db",
}


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return False
    rel_posix = rel.as_posix()
    if rel_posix in EXCLUDE_FILES:
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    if path.name.startswith(".env") and path.name != ".env.example":
        return False
    return True


def main() -> None:
    files = sorted(
        p for p in ROOT.rglob("*") if p.is_file() and should_include(p)
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            zf.write(path, f"{PREFIX}/{rel}")

    size_mb = OUT.stat().st_size / 1e6
    print(f"Created {OUT} ({len(files)} files, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
