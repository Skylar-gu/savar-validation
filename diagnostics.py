"""Lightweight run diagnostics — one-line breadcrumbs so cluster/GPU runs are
diagnosable after the fact instead of failing silently.

Emit a structured context line at the start of a run (dataset/checkpoint/output
paths, device, CUDA status, git commit) and a failure line on error:

    from diagnostics import run_context, run_failure
    run_context(stage="cnn_train", dataset="data/splits_diurnal",
                checkpoint="checkpoints_diurnal/best.pt")
    try:
        ...
    except Exception as e:
        run_failure("cnn_train", e); raise
"""
import datetime
import json
import subprocess


def cuda_status():
    """{'cuda': bool|None, 'device': str, ...} without requiring torch."""
    try:
        import torch
    except Exception as e:  # torch not installed
        return {"cuda": None, "device": "unknown", "error": repr(e)}
    if torch.cuda.is_available():
        return {"cuda": True,
                "device": torch.cuda.get_device_name(0),
                "n_gpu": torch.cuda.device_count()}
    return {"cuda": False, "device": "cpu"}


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def run_context(**fields):
    """Print and return a run-context breadcrumb. Pass any run-specific fields
    (stage, dataset, checkpoint, output, model_id, ...)."""
    ctx = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
           "git": git_commit(),
           **cuda_status(),
           **fields}
    print("[runctx] " + json.dumps(ctx, default=str), flush=True)
    return ctx


def run_failure(stage, error):
    """Print a structured failure breadcrumb (call before re-raising)."""
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
           "stage": stage,
           "error_type": type(error).__name__,
           "error": str(error)}
    print("[runfail] " + json.dumps(rec, default=str), flush=True)
    return rec
