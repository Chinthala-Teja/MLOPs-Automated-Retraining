"""setup_env.py
Writes retrainiq.pth into the active Python environment's site-packages (purelib)
pointing to the absolute paths of the component folders so all ri_* modules can be imported directly.
"""
from pathlib import Path
import sys
import sysconfig

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folders = [
        root / "0-Init",
        root / "1-Profile",
        root / "2-Drift",
        root / "3-CARA",
        root / "4-Metrics",
        root / "5-Analysis",
        root / "6-Decision",
        root / "frontend",
    ]

    purelib = sysconfig.get_path("purelib")
    if not purelib:
        print("Error: Could not locate purelib in current Python environment.", file=sys.stderr)
        sys.exit(1)

    site_packages = Path(purelib)
    site_packages.mkdir(parents=True, exist_ok=True)
    pth_file = site_packages / "retrainiq.pth"

    content = "\n".join(str(f.resolve()) for f in folders) + "\n"
    pth_file.write_text(content, encoding="utf-8")
    print(f"Wrote {pth_file} containing:")
    print(content)

if __name__ == "__main__":
    main()

