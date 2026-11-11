import pytest
import time
from ri_init.worker import run_in_worker, WorkerError

def sample_add(a: int, b: int) -> dict:
    return {"sum": a + b}

def sample_sleep(seconds: float) -> dict:
    time.sleep(seconds)
    return {"slept": seconds}

def sample_fail() -> dict:
    raise ValueError("Deliberate failure")

def test_worker_trivial():
    res = run_in_worker("tests.test_worker:sample_add", {"a": 3, "b": 7}, timeout=10)
    assert res == {"sum": 10}

def test_worker_timeout():
    with pytest.raises(WorkerError) as exc_info:
        run_in_worker("tests.test_worker:sample_sleep", {"seconds": 2.0}, timeout=1)
    assert "timed out" in str(exc_info.value).lower()

def test_worker_error():
    with pytest.raises(WorkerError) as exc_info:
        run_in_worker("tests.test_worker:sample_fail", {}, timeout=10)
    assert "Deliberate failure" in str(exc_info.value) or "ValueError" in str(exc_info.value)

