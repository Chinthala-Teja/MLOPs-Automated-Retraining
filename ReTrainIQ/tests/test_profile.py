from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from ri_init.config import WORKSPACE
import ri_init.store as store
from ri_analysis.llm import MockLLM
from ri_profile.understand import (
    extract_profile,
    extract_zip_safe,
    fetch_repo,
    finalize_profile,
    heuristic_profile,
    verify_profile,
)


def test_demo_profile_verifies_cleanly():
    demo_profile_path = WORKSPACE / "demo" / "profile.json"
    assert demo_profile_path.exists(), "workspace/demo/profile.json must exist"

    with open(demo_profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    issues = verify_profile(profile)
    errors = [iss for iss in issues if iss.startswith("ERROR:")]
    assert not errors, f"Expected no ERRORs on demo profile, got: {errors}"


def test_zip_escape_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "evil.zip"
        dest_dir = Path(tmpdir) / "dest"

        # Create zip with escape
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../escape.txt", "malicious payload")

        with pytest.raises(ValueError) as exc:
            extract_zip_safe(zip_path, dest_dir)
        assert "escape" in str(exc.value).lower() or "traversal" in str(exc.value).lower()


def test_zip_symlink_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "symlink.zip"
        dest_dir = Path(tmpdir) / "dest"

        # Create zip with symlink entry (Unix mode 0o120000)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zinfo = zipfile.ZipInfo("symlink_entry")
            zinfo.create_system = 3  # Unix
            zinfo.external_attr = 0o120777 << 16
            zf.writestr(zinfo, "/etc/passwd")

        with pytest.raises(ValueError) as exc:
            extract_zip_safe(zip_path, dest_dir)
        assert "symlink" in str(exc.value).lower()


def test_fetch_repo_url_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_dir = Path(tmpdir) / "repo"

        # Non-GitHub URL
        with pytest.raises(ValueError) as exc1:
            fetch_repo("https://gitlab.com/user/repo.git", dest_dir)
        assert "Invalid GitHub" in str(exc1.value)

        # Extra path segments
        with pytest.raises(ValueError) as exc2:
            fetch_repo("https://github.com/user/repo/extra/segment", dest_dir)
        assert "Invalid GitHub" in str(exc2.value)


def test_verify_profile_missing_column_and_missing_model():
    demo_profile_path = WORKSPACE / "demo" / "profile.json"
    with open(demo_profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # Missing column
    bad_col_profile = json.loads(json.dumps(profile))
    bad_col_profile["features"]["numeric"].append("nonexistent_column_xyz")
    issues_col = verify_profile(bad_col_profile)
    assert any("ERROR: Feature column 'nonexistent_column_xyz' not found" in iss for iss in issues_col)

    # Missing model file
    bad_model_profile = json.loads(json.dumps(profile))
    bad_model_profile["model"]["artifact"] = "workspace/demo/models/v1/missing_model.joblib"
    issues_model = verify_profile(bad_model_profile)
    assert any("ERROR: Model artifact file does not exist" in iss for iss in issues_model)


def test_extract_profile_with_mock_llm_and_retry_failure():
    # Valid profile payload
    valid_payload = {
        "project_id": "test_proj",
        "problem_statement": "Predict delays",
        "task": {"type": "classification", "subtype": "binary", "target": "delayed", "positive_class": 1},
        "features": {"numeric": ["dep_delay"], "categorical": ["carrier"]},
        "model": {"framework": "sklearn", "artifact": "workspace/demo/models/v1/model.joblib", "algorithm": "RF"},
        "data": {"reference_path": "workspace/demo/data/reference.csv", "holdout_path": None, "id_column": None, "ordered": True},
        "primary_metric": "recall",
        "batch_size_B": 2000,
        "offline_batches_N": 25,
        "W": 1,
        "seed": 42,
    }

    mock_llm_ok = MockLLM(json_response=valid_payload)
    res = extract_profile("sample context", mock_llm_ok)
    assert res["project_id"] == "test_proj"

    # Malformed payload that fails twice
    mock_llm_bad = MockLLM(json_response={"invalid": "payload"})
    with pytest.raises(ValueError) as exc:
        extract_profile("sample context", mock_llm_bad)
    assert "Failed to extract valid profile" in str(exc.value)


def test_heuristic_profile_batch_reduction():
    ref_path = WORKSPACE / "demo" / "data" / "reference.csv"
    assert ref_path.exists()

    # On reference (50,000 rows): B should be 2000
    prof_full = heuristic_profile(dataset_path=ref_path)
    assert prof_full["batch_size_B"] == 2000

    # On 6,000 rows: B should be reduced (6000 // 25 = 240 -> max(500, 240) = 500)
    holdout_path = WORKSPACE / "demo" / "data" / "holdout.csv"
    prof_small = heuristic_profile(dataset_path=holdout_path)
    assert prof_small["batch_size_B"] == 500


def test_finalize_profile_writes_and_stores():
    demo_profile_path = WORKSPACE / "demo" / "profile.json"
    with open(demo_profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    profile["project_id"] = "final_test_proj"
    profile["source"] = "test_unit"

    final = finalize_profile(profile, user_edits={"problem_statement": "Custom edited problem statement"})

    assert final["problem_statement"] == "Custom edited problem statement"

    # Verify profile.json written
    saved_file = WORKSPACE / "final_test_proj" / "profile.json"
    assert saved_file.exists()

    with open(saved_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["problem_statement"] == "Custom edited problem statement"

    # Verify registered in SQLite
    proj_row = store.get_project("final_test_proj")
    assert proj_row is not None
    assert proj_row["id"] == "final_test_proj"

