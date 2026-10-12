"""Portable unit-test boundary for nuPlan's POSIX-only file-lock import.

The core tensor/lifecycle tests never execute GPKG locking. Real map-cache
integration remains a Linux test; attempting it here must fail explicitly.
"""

import importlib.util
import os
from pathlib import Path
import shutil
import sys
import types

import pytest


if os.name == "nt" and importlib.util.find_spec("fcntl") is None:
    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_SH = 1
    fcntl.LOCK_EX = 2
    fcntl.LOCK_NB = 4
    fcntl.LOCK_UN = 8

    def _unsupported_flock(*args, **kwargs):
        raise OSError("nuPlan GPKG file locking requires a POSIX integration environment.")

    fcntl.flock = _unsupported_flock
    sys.modules["fcntl"] = fcntl


@pytest.fixture
def bash_executable():
    if os.name == "nt":
        git = shutil.which("git")
        candidates = (
            [Path(git).resolve().parents[1] / suffix for suffix in ("bin/bash.exe", "usr/bin/bash.exe")]
            if git else []
        )
        executable = next((str(path) for path in candidates if path.is_file()), None)
    else:
        executable = shutil.which("bash")
    if executable is None:
        pytest.skip("Bash is required for shell entrypoint checks (Git Bash on Windows).")
    return executable
