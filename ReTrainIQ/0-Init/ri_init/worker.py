"""worker.py
Sandbox runner for isolated execution.
"""
from __future__ import annotations

import os
import sys
import tempfile
import pickle
import subprocess
from pathlib import Path

class WorkerError(Exception):
    """Raised when worker subprocess execution fails or times out."""
    pass

def run_in_worker(callable_path: str, kwargs: dict, timeout: int = 900) -> dict:
    """
    Run sys.executable -m ri_init.worker_main as a subprocess,
    pass kwargs and read result through temp pickle files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.pkl"
        output_file = Path(tmpdir) / "output.pkl"

        with open(input_file, "wb") as f:
            pickle.dump(kwargs, f)

        cmd = [
            sys.executable,
            "-m",
            "ri_init.worker_main",
            callable_path,
            str(input_file),
            str(output_file),
        ]

        # Minimal environment with PYTHONPATH
        env = {}
        for var in ["PATH", "SYSTEMROOT", "VIRTUAL_ENV", "TEMP", "TMP"]:
            if var in os.environ:
                env[var] = os.environ[var]

        # Ensure PYTHONPATH contains current sys.path entries
        pythonpath = os.pathsep.join(sys.path)
        if "PYTHONPATH" in os.environ:
            pythonpath = os.environ["PYTHONPATH"] + os.pathsep + pythonpath
        env["PYTHONPATH"] = pythonpath

        # Spawn subprocess with Popen to guarantee kill on timeout
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            raise WorkerError(f"Worker timed out after {timeout} seconds")

        if proc.returncode != 0:
            raise WorkerError(f"Worker failed with exit code {proc.returncode}:\n{stderr}")

        if not output_file.exists():
            raise WorkerError(f"Worker completed without writing output file.\nStderr:\n{stderr}")

        with open(output_file, "rb") as f:
            result = pickle.load(f)

        return result
