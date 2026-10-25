from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency in this project
    torch = None

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed python's random, numpy, and torch (CPU + all CUDA devices) in one call.

    deterministic=True additionally forces cuDNN into deterministic mode.
    This makes runs bit-reproducible at some training-speed cost — turn it on
    for final paper-reported runs, leave it off for quick iteration.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

def resolve_device(prefer: str = "cuda") -> "torch.device":
    """Return the best available torch.device, with a clear log message about the choice."""
    if torch is None:
        raise RuntimeError("torch is not installed in this environment")
    if prefer == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return torch.device(f"cuda:{idx}")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False

def get_logger(name: str, level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Returns a module-level logger with a consistent format across the whole
    project. Safe to call repeatedly (won't duplicate handlers).
    """
    global _configured
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(level)
        _configured = True

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(file_handler)

    return logger

def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown (not a git repo, or git unavailable)"

def write_run_metadata(output_dir: Path, config: dict, extra: Optional[dict] = None) -> Path:
    """
    Writes a run_metadata.json next to a run's outputs, capturing the git
    commit, timestamp, config dict, and python/torch versions. Call this once
    at the start of train.py / evaluate.py / explain.py for every protocol run
    so results are always traceable back to the exact code + config that
    produced them — this is the kind of thing reviewers and future
    reproducers specifically look for.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "python_version": sys.version,
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "cuda_available": bool(torch.cuda.is_available()) if torch is not None else None,
        "config": config,
    }
    if extra:
        meta.update(extra)

    path = output_dir / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path

def cli_entrypoint(main_fn: Callable[[], Optional[int]]) -> None:
    """
    Call this at module level, once, in place of the repeated

        if __name__ == "__main__":
            main()

    block found in almost every script in this project. Usage:

        def main():
            ...

        cli_entrypoint(main)

    It still only actually runs `main_fn()` when the file is executed as a
    script (not when imported), so behavior is identical to the pattern it
    replaces — it just removes fifteen copies of the same two lines and adds
    a consistent top-level error message if main() raises.
    """
    frame = sys._getframe(1)
    if frame.f_globals.get("__name__") != "__main__":
        return

    try:
        exit_code = main_fn()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        logging.getLogger(main_fn.__module__).exception("Unhandled exception in main()")
        sys.exit(1)
    sys.exit(exit_code if isinstance(exit_code, int) else 0)
