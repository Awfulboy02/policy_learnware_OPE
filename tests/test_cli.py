from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from policy_learnware_ope.adapters import RAW_QUERY_SCHEMA
from policy_learnware_ope.benchmark import load_ranking_seal
from policy_learnware_ope.cli import _raw_membership_digest, _toy_batch, main, run_toy


EXPECTED_METHODS = {
    "FH_FQE_G099_H5",
    "FH_KMIFQE_G099_H5",
    "ETM_MBOPE_G099_H5",
    "DOPE_STYLE_MB_FF_G099_H5",
    "AR_MBOPE_G099_H5",
    "RAW_ADAPTER_FIXTURE",
}


def test_toy_runner_fits_all_methods_seals_then_exports_metrics(tmp_path: Path):
    output = tmp_path / "toy"
    result = run_toy(output, seed=17, implementation_commit="c" * 40)

    assert result["status"] == "TOY_MVP_PASS"
    assert result["seed"] == 17
    assert result["implementation_commit"] == "c" * 40
    assert result["implementation"] == {
        "commit": "c" * 40,
        "tree": "CALLER_SUPPLIED",
        "worktree_status": "CALLER_SUPPLIED",
    }
    assert len(result["config_sha256"]) == 64
    assert set(result["method_scope"]) == EXPECTED_METHODS
    assert all("H1000" not in method_id for method_id in result["method_scope"])
    assert result["real_asset_training_started"] is False
    assert result["production_raw_status"] == "NO_GO_RAW_OPERATOR_AUTHORITY"
    assert len(result["metrics"]) == len(EXPECTED_METHODS)
    for method_id, seal_ref in result["ranking_seals"].items():
        seal_path = output / seal_ref["path"]
        seal = load_ranking_seal(
            seal_path,
            expected_seal_digest=seal_ref["sha256"],
        )
        assert seal.payload["method_id"] == method_id
        assert seal.payload["value_convention"] == "J_gamma=0.99_H=5_raw"
        assert len(seal.payload["ranking"]) == 5
        sealed_text = seal_path.read_text(encoding="utf-8").casefold()
        assert "oracle" not in sealed_text
        assert "runtime" not in sealed_text
    raw_metrics = next(
        row for row in result["metrics"] if row["method_id"] == "RAW_ADAPTER_FIXTURE"
    )
    assert raw_metrics["value_mae"] is None
    assert raw_metrics["value_rmse"] is None
    assert raw_metrics["runtime_seconds"] >= 0.0
    for method_id, rows in result["estimates"].items():
        assert len(rows) == 5
        assert {row["status"] for row in rows.values()} == {"PASS"}, method_id
        assert {row["method_id"] for row in rows.values()} == {method_id}
    assert (output / result["artifacts"]["metrics_json"]).is_file()
    assert (output / result["artifacts"]["metrics_csv"]).is_file()
    assert (output / result["artifacts"]["runtime"]).is_file()
    query = json.loads((output / result["artifacts"]["raw_query"]).read_text())
    assert query["schema"] == RAW_QUERY_SCHEMA
    assert set(query["fields"]) == {
        "observation",
        "action",
        "next_observation",
        "native_timestep",
        "episode_offsets",
    }
    run_text = (output / "run.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in run_text


def test_same_seed_has_stable_semantics_and_byte_identical_seals(tmp_path: Path):
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first = run_toy(first_output, seed=23, implementation_commit="d" * 40)
    second = run_toy(second_output, seed=23, implementation_commit="d" * 40)

    assert first["reproducibility_sha256"] == second["reproducibility_sha256"]
    assert first["estimates"] == second["estimates"]
    assert first["raw_scores"] == second["raw_scores"]
    for method_id in EXPECTED_METHODS:
        first_ref = first["ranking_seals"][method_id]
        second_ref = second["ranking_seals"][method_id]
        assert first_ref["sha256"] == second_ref["sha256"]
        assert (first_output / first_ref["path"]).read_bytes() == (
            second_output / second_ref["path"]
        ).read_bytes()


def test_raw_membership_binds_reward_free_physical_rows() -> None:
    batch = _toy_batch(31, episodes=2)
    reward_changed = replace(batch, reward=batch.reward + 100.0)
    next_state_changed = replace(
        batch,
        next_observation=batch.next_observation + 0.25,
    )

    assert _raw_membership_digest(reward_changed) == _raw_membership_digest(batch)
    assert _raw_membership_digest(next_state_changed) != _raw_membership_digest(batch)


def test_cli_smoke_prints_machine_readable_summary(tmp_path: Path, capsys):
    output = tmp_path / "cli-toy"
    assert main(["toy", "--output", str(output), "--seed", "19"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "TOY_MVP_PASS"
    assert set(printed["methods"]) == EXPECTED_METHODS
    assert Path(printed["run"]).is_file()
    assert len(printed["run_sha256"]) == 64
    assert set(printed["ranking_seal_sha256"]) == EXPECTED_METHODS
    assert len(printed["oracle_manifest_sha256"]) == 64


def test_toy_runner_refuses_nonempty_output_without_partial_overwrite(tmp_path: Path):
    output = tmp_path / "no-clobber"
    run_toy(output, seed=5, implementation_commit="a" * 40)
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}

    with pytest.raises(FileExistsError, match="refusing partial overwrite"):
        run_toy(output, seed=6, implementation_commit="b" * 40)

    after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert after == before


def test_real_preflight_is_stable_and_all_production_gates_are_no_go(
    tmp_path: Path,
    capsys,
):
    first = tmp_path / "preflight-a.json"
    second = tmp_path / "preflight-b.json"
    assert main(["real-preflight", "--output", str(first)]) == 2
    capsys.readouterr()
    assert main(["real-preflight", "--output", str(second)]) == 2
    capsys.readouterr()

    assert first.read_bytes() == second.read_bytes()
    report = json.loads(first.read_text(encoding="utf-8"))
    assert report["status"] == "NO_GO"
    assert report["production_training_started"] is False
    assert report["asset_mutation_started"] is False
    assert set(report["required_gates"]) == {
        "actor_authority",
        "discounted_oracle",
        "exact_behavior_density",
    }
    assert {gate["status"] for gate in report["required_gates"].values()} == {"NO_GO"}
    assert report["raw_adapter"]["status"] == "NO_GO"


def test_census_cli_missing_dataset_emits_stable_reproduction_metadata(
    tmp_path: Path,
    capsys,
):
    output = tmp_path / "census.json"
    missing = tmp_path / "missing.npz"
    assert main(
        [
            "census",
            "--dataset",
            str(missing),
            "--horizon",
            "1000",
            "--output",
            str(output),
        ]
    ) == 2
    capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "NO_GO"
    assert report["seed"] is None
    assert report["config"]["asset_mode"] == "READ_ONLY"
    assert len(report["config_sha256"]) == 64
    assert set(report["implementation"]) == {"commit", "tree", "worktree_status"}
    assert report["provenance"]["input_paths_recorded"] is False
    assert str(tmp_path) not in output.read_text(encoding="utf-8")


def test_cli_outputs_are_no_clobber_and_never_enter_another_git_repo(
    tmp_path: Path,
):
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"do-not-overwrite\n")
    with pytest.raises(FileExistsError):
        main(["real-preflight", "--output", str(existing)])
    assert existing.read_bytes() == b"do-not-overwrite\n"

    foreign_repo = tmp_path / "frozen-old-repo"
    (foreign_repo / ".git").mkdir(parents=True)
    forbidden = foreign_repo / "artifacts" / "toy"
    with pytest.raises(PermissionError, match="different Git repository"):
        run_toy(forbidden, seed=1, implementation_commit="f" * 40)
    assert not forbidden.exists()
