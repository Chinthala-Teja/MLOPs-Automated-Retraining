import pytest
import tempfile
from pathlib import Path
from ri_init.store import (
    init_db,
    add_project,
    get_project,
    add_batch,
    add_report,
    get_reports,
    add_decision,
    add_version,
    set_active_version,
    get_active_version,
)

@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_db(db_path)
        yield db_path

def test_store_roundtrip(temp_db):
    # 1. Project
    add_project("p1", "source_git", '{"name":"test"}', "active", db_path=temp_db)
    p = get_project("p1", db_path=temp_db)
    assert p is not None
    assert p["id"] == "p1"
    assert p["source"] == "source_git"

    # 2. Batch
    b_id = add_batch("p1", "batch1.csv", 100, 50, db_path=temp_db)
    assert b_id == 1

    # 3. Report
    r_id = add_report("p1", b_id, "{}", "{}", "{}", "{}", "KEEP", "All good", db_path=temp_db)
    assert r_id == 1
    reports = get_reports("p1", db_path=temp_db)
    assert len(reports) == 1
    assert reports[0]["zone"] == "KEEP"

    # 4. Decision
    d_id = add_decision(r_id, "KEEP", "system", "Auto keep", db_path=temp_db)
    assert d_id == 1

    # 5. Model version
    v1_id = add_version("p1", "v1", "path/to/v1", "batch1", "active", db_path=temp_db)
    assert v1_id == 1
    active = get_active_version("p1", db_path=temp_db)
    assert active is not None
    assert active["version"] == "v1"

    v2_id = add_version("p1", "v2", "path/to/v2", "batch2", "candidate", db_path=temp_db)
    set_active_version("p1", "v2", db_path=temp_db)
    active2 = get_active_version("p1", db_path=temp_db)
    assert active2["version"] == "v2"

