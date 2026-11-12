from __future__ import annotations

from pathlib import Path
import subprocess
import sys

# Ensure root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.status import check_status  # noqa: E402

def test_status_tool_passes():
    rows, errors = check_status()
    assert not errors, f"status.py reported contract/signature errors: {errors}"
    assert len(rows) > 0

def test_status_cli_exit_code():
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "status.py")], capture_output=True, text=True)
    assert res.returncode == 0
    assert "All CONTRACT.md signatures and modules verified successfully." in res.stdout
