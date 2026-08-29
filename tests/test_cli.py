from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

import policy_learnware_ope.cli as cli_module
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
    assert main(
        [
            "toy",
            "--artifacts-root",
            str(tmp_path),
            "--output",
            "cli-toy",
            "--seed",
            "19",
        ]
    ) == 0
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


def test_toy_direct_api_resolves_relative_roots_and_preserves_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    relative = Path("ope/toy-existing")
    cases = [
        (tmp_path / "environment", tmp_path / "environment", None),
        (tmp_path / "explicit", tmp_path / "ignored-env", tmp_path / "explicit"),
    ]
    for checkout, environment_root, explicit_root in cases:
        if environment_root is None:
            monkeypatch.delenv("RL_LEARNWARE_ARTIFACTS_ROOT", raising=False)
            monkeypatch.setattr(
                cli_module,
                "_verified_source_checkout",
                lambda checkout=checkout: checkout,
            )
            expected = checkout.parent / "artifacts" / relative
        else:
            monkeypatch.setenv("RL_LEARNWARE_ARTIFACTS_ROOT", str(environment_root))
            expected_root = explicit_root or environment_root
            expected = expected_root / relative
        expected.mkdir(parents=True)
        marker = expected / "keep.txt"
        marker.write_text("do-not-overwrite\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="refusing partial overwrite"):
            run_toy(
                relative,
                seed=3,
                implementation_commit="c" * 40,
                artifacts_root=explicit_root,
            )
        assert marker.read_text(encoding="utf-8") == "do-not-overwrite\n"


def test_real_preflight_is_stable_and_all_production_gates_are_no_go(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
):
    first = tmp_path / "preflight-a.json"
    second = tmp_path / "preflight-b.json"
    monkeypatch.setenv("RL_LEARNWARE_ARTIFACTS_ROOT", str(tmp_path))
    assert main(["real-preflight", "--output", first.name]) == 2
    capsys.readouterr()
    assert main(["real-preflight", "--output", second.name]) == 2
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
            "--artifacts-root",
            str(tmp_path),
            "--dataset",
            missing.name,
            "--horizon",
            "1000",
            "--output",
            output.name,
        ]
    ) == 2
    capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "NO_GO"
    assert report["seed"] is None
    assert report["config"]["asset_mode"] == "READ_ONLY"
    assert len(report["config_sha256"]) == 64
    identity_keys = set(report["implementation"])
    required_identity = {"commit", "tree", "worktree_status"}
    assert required_identity <= identity_keys
    if cli_module._verified_source_checkout() is not None:
        assert identity_keys == required_identity
    else:
        assert {"package_name", "package_version"} <= identity_keys
    assert report["provenance"]["input_paths_recorded"] is False
    assert str(tmp_path) not in output.read_text(encoding="utf-8")


def test_cli_outputs_are_no_clobber_and_never_enter_a_git_repo(
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
    with pytest.raises(PermissionError, match="refusing to write into a Git repository"):
        run_toy(forbidden, seed=1, implementation_commit="f" * 40)
    assert not forbidden.exists()

    source_root = cli_module._verified_source_checkout()
    if source_root is not None:
        with pytest.raises(
            PermissionError, match="refusing to write into a Git repository"
        ):
            cli_module._guard_output_location(source_root / "artifacts" / "forbidden")


def test_artifact_root_resolution_order_and_safe_source_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_root = cli_module._verified_source_checkout()
    monkeypatch.delenv("RL_LEARNWARE_ARTIFACTS_ROOT", raising=False)
    if source_root is None:
        with pytest.raises(ValueError, match="relative artifact path requires"):
            cli_module._resolve_artifact_path("ope/default.json")
    else:
        with pytest.raises(ValueError, match="published relocation manifest"):
            cli_module._resolve_artifact_path("ope/default.json")

    environment_root = tmp_path / "environment"
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv("RL_LEARNWARE_ARTIFACTS_ROOT", str(environment_root))
    assert cli_module._resolve_artifact_path("ope/env.json") == (
        environment_root / "ope" / "env.json"
    ).resolve()
    assert cli_module._resolve_artifact_path(
        "ope/explicit.json", artifacts_root=explicit_root
    ) == (explicit_root / "ope" / "explicit.json").resolve()

    absolute = tmp_path / "absolute.json"
    assert cli_module._resolve_artifact_path(
        absolute, artifacts_root=explicit_root
    ) == absolute.resolve()

    with pytest.raises(ValueError, match="escapes artifacts root"):
        cli_module._resolve_artifact_path(
            "../outside.json", artifacts_root=explicit_root
        )
    explicit_root.mkdir()
    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    (explicit_root / "escape").symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        cli_module._resolve_artifact_path(
            "escape/outside.json", artifacts_root=explicit_root
        )


def test_artifact_resolver_rejects_symlink_roots_empty_env_and_git_env_spoof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        cli_module._resolve_artifact_path("x", artifacts_root=alias)
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink component"):
        cli_module._resolve_artifact_path(broken / "x")
    monkeypatch.setenv(cli_module.ARTIFACTS_ROOT_ENV, "  ")
    with pytest.raises(ValueError, match="must not be empty"):
        cli_module._resolve_artifact_path("x")
    monkeypatch.setenv("GIT_DIR", str(Path.cwd() / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    monkeypatch.setenv("git_dir", str(tmp_path / "lowercase-spoof"))
    monkeypatch.setenv("PATH", str(tmp_path / "fake-bin"))
    assert cli_module._verified_source_checkout() is None


def test_installed_layout_requires_root_and_rejects_foreign_git_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    foreign = tmp_path / "foreign-consumer"
    (foreign / ".git").mkdir(parents=True)
    installed_package = foreign / "site" / "policy_learnware_ope"
    shutil.copytree(
        Path(cli_module.__file__).resolve().parent,
        installed_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    monkeypatch.setattr(cli_module, "__file__", str(installed_package / "cli.py"))
    monkeypatch.setattr(cli_module, "distribution_version", lambda _name: "0.4.0b0")
    monkeypatch.delenv("RL_LEARNWARE_ARTIFACTS_ROOT", raising=False)

    assert cli_module._verified_source_checkout() is None
    with pytest.raises(ValueError, match="relative artifact path requires"):
        cli_module._resolve_artifact_path("ope/toy")

    identity = cli_module._implementation_identity()
    assert identity["commit"].startswith("PACKAGE_CONTENT_SHA256:")
    assert identity["worktree_status"] == "INSTALLED_IMMUTABLE_CONTENT"
    with pytest.raises(PermissionError, match="refusing to write into a Git repository"):
        cli_module._guard_output_location(foreign / "artifacts" / "toy")

    external_root = tmp_path / "artifacts"
    external = cli_module._resolve_artifact_path(
        "ope/toy", artifacts_root=external_root
    )
    assert external == (external_root / "ope" / "toy").resolve()
    assert cli_module._guard_output_location(external) == external
