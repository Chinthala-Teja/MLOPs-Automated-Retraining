"""worker_main.py
Subprocess runner for isolated callable execution.
Imports 'package.module:function', loads kwargs from pickle, runs, and pickles the result.
"""
import sys
import pickle
import importlib
from pathlib import Path

def setup_limits() -> None:
    try:
        import resource
        # 4 GB address space limit
        limit_bytes = 4 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        # CPU time limit e.g. 900 seconds
        resource.setrlimit(resource.RLIMIT_CPU, (900, 950))
    except (ImportError, AttributeError, ValueError):
        # On Windows or unsupported environments, rely on subprocess timeout
        pass

def main() -> None:
    if len(sys.argv) < 4:
        sys.stderr.write("Usage: python -m ri_init.worker_main <callable_path> <input_pickle> <output_pickle>\n")
        sys.exit(1)

    callable_path = sys.argv[1]
    input_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3])

    setup_limits()

    if ":" not in callable_path:
        sys.stderr.write(f"Invalid callable_path format '{callable_path}'. Expected 'module:function'.\n")
        sys.exit(1)

    mod_name, func_name = callable_path.split(":", 1)

    try:
        with open(input_file, "rb") as f:
            kwargs = pickle.load(f)

        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name)
        result = func(**kwargs)

        with open(output_file, "wb") as f:
            pickle.dump(result, f)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()

