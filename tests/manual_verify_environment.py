"""Manual verification script — package versions + import compatibility, nothing else.

Not a pytest-collected test. Two simple checks, in order:
1. Print the installed version of every package pyproject.toml declares dscompanion needs
   (read directly from `[project.dependencies]`, so this never drifts out of sync
   with the actual dependency list).
2. Import every module under dscompanion/ and report which ones fail — the fastest way
   to know whether the packages installed on this machine/cluster are actually
   compatible with dscompanion's code, without needing any data or a pipeline run.

Run with (from dscompanion/ as cwd)::

    conda activate dscompanion312 && python tests/manual_verify_environment.py
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def dependency_names() -> list[str]:
    """Read package names out of pyproject.toml's [project.dependencies].

    Args:
        None

    Returns:
        list[str]: Distribution names (e.g. "scikit-learn"), version pins stripped.
    """
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    raw = data["project"]["dependencies"]
    return [re.split(r"[<>=!~]", dep, maxsplit=1)[0].strip() for dep in raw]


def print_package_versions() -> None:
    """Print Python + every pyproject.toml dependency's installed version.

    Args:
        None

    Returns:
        None
    """
    print(f"Python: {sys.version.split()[0]} ({sys.executable})\n")
    for name in dependency_names():
        try:
            print(f"  {name}=={version(name)}")
        except PackageNotFoundError:
            print(f"  {name}==NOT INSTALLED")


def import_all_dscompanion_modules() -> list[tuple[str, str]]:
    """Import every submodule under dscompanion/ and report which ones fail.

    Args:
        None

    Returns:
        list[tuple[str, str]]: (module_name, error_message) for each import
        that raised — empty list if everything imported cleanly.
    """
    import dscompanion

    failures: list[tuple[str, str]] = []
    modules = [
        mod.name for mod in pkgutil.walk_packages(dscompanion.__path__, prefix="dscompanion.")
    ]
    print(f"\nImporting {len(modules) + 1} modules under dscompanion/ ...")
    for name in ["dscompanion", *modules]:
        try:
            importlib.import_module(name)
            print(f"  OK   {name}")
        except Exception as exc:
            print(f"  FAIL {name} — {type(exc).__name__}: {exc}")
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    return failures


def main() -> None:
    print_package_versions()
    failures = import_all_dscompanion_modules()

    if failures:
        print(f"\n{len(failures)} module(s) failed to import:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        sys.exit(1)

    print("\nALL MODULES IMPORTED CLEANLY.")


if __name__ == "__main__":
    main()
