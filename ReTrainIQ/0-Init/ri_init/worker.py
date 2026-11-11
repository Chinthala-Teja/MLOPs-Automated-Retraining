"""worker.py
Sandbox runner for isolated execution.
"""
import os
import sys
import tempfile
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict

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

        # Minimal environment
        env = {}
        if "PATH" in os.environ:
            env["PATH"] = os.environ["PATH"]
        if "SYSTEMROOT" in os.environ:
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        if "PYTHONPATH" in os.environ:
            env["PYTHONPATH"] = os.environ["PYTHONPATH"]
        if "VIRTUAL_ENV" in os.environ:
            env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
        if "TEMP" in os.environ:
            env["TEMP"] = os.environ["TEMP"]
        if "TMP" in os.environ:
            env["TMP"] = os.environ["TMP"]

        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise WorkerError(f"Worker timed out after {timeout} seconds: {e}")

        if proc.returncode != 0:
            raise WorkerError(f"Worker failed with exit code {proc.returncode}:\n{proc.stderr}")

        if not output_file.exists():
            raise WorkerError(f"Worker completed without writing output file.\nStderr:\n{proc.stderr}")

        with open(output_file, "rb") as f:
            result = pickle.load(f)

        return result
