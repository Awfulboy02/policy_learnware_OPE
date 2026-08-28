from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import policy_learnware_ope.cli as cli_module
from policy_learnware_ope.adapters import (
    GateClosed,
    RAW_PROJECT_METHOD_ID,
    RAW_QUERY_SCHEMA,
    RAW_RESPONSE_SCHEMA,
    RAW_SCORE_SEMANTICS,
    sha256_file,
)
from policy_learnware_ope.benchmark import load_ranking_seal
from policy_learnware_ope.cli import (
    REAL_SMOKE_CONFIG_SCHEMA,
    _raw_membership_digest,
    _toy_batch,
    main,
    run_real_smoke,
    run_toy,
)
from policy_learnware_ope.core import (
    DataValidationError,
    EstimateStatus,
    PolicySemantics,
    ValueEstimate,
)


EXPECTED_METHODS = {
    "FH_FQE_G099_H5",
    "FH_KMIFQE_G099_H5",
    "ETM_MBOPE_G099_H5",
    "DOPE_STYLE_MB_FF_G099_H5",
    "AR_MBOPE_G099_H5",
    "RAW_ADAPTER_FIXTURE",
}


def test_toy_runner_fits_all_methods_seals_then_exports_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "toy"
    original_oracle = cli_module._toy_oracle
    original_sample_actions = cli_module._ToyActor.sample_actions
    action_key_ledgers: dict[
        str, list[tuple[tuple[int, ...], bytes, tuple[int, ...], bytes]]
    ] = {}

    def record_action_keys(self, observations, native_timestep, *, keys):
        times = np.asarray(native_timestep)
        key_array = np.asarray(keys)
        action_key_ledgers.setdefault(self.policy_id, []).append(
            (times.shape, times.tobytes(), key_array.shape, key_array.tobytes())
        )
        return original_sample_actions(
            self, observations, native_timestep, keys=keys
        )

    monkeypatch.setattr(cli_module._ToyActor, "sample_actions", record_action_keys)

    def oracle_after_all_seals(*args, **kwargs):
        assert {
            path.stem for path in (output / "seals").glob("*.json")
        } == EXPECTED_METHODS
        return original_oracle(*args, **kwargs)

    monkeypatch.setattr(cli_module, "_toy_oracle", oracle_after_all_seals)
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
    assert "candidate_id" not in result["config"]["common_random_key_derivation"]
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
        for candidate_id, row in rows.items():
            assert row["provenance"]["candidate_id"] == candidate_id
            assert row["provenance"]["value_convention"] == "J_gamma=0.99_H=5_raw"
            assert row["cost"]["fit_transitions"] > 0
            assert row["cost"]["timing_artifact"] == "runtime.json"
            assert not any(key.endswith("_seconds") for key in row["cost"])
    fqe_rows = result["estimates"]["FH_FQE_G099_H5"]
    for field in (
        "fit_key_digest",
        "fit_next_action_key_schedule_digest",
        "fit_support_action_key_schedule_digest",
        "fit_action_key_schedule_digest",
        "estimate_key_digest",
        "estimate_action_key_schedule_digest",
    ):
        assert len({row["provenance"][field] for row in fqe_rows.values()}) == 1
    assert all(row["cost"]["actor_query_rows"] > 0 for row in fqe_rows.values())
    for method_id in (
        "ETM_MBOPE_G099_H5",
        "DOPE_STYLE_MB_FF_G099_H5",
        "AR_MBOPE_G099_H5",
    ):
        rows = result["estimates"][method_id]
        assert len(
            {row["diagnostics"]["rollout_key_digest"] for row in rows.values()}
        ) == 1
        assert len({row["provenance"]["fit_key_digest"] for row in rows.values()}) == 1
        assert all(row["cost"]["actor_queries"] > 0 for row in rows.values())
    # This is the actual actor-call ledger across FQE/KMIFQE, all three model
    # rollouts, and the post-seal oracle.  Candidate-specific observations may
    # differ, but phase/draw-aligned native times and keys must be identical.
    assert set(action_key_ledgers) == {
        f"toy-policy-{index}" for index in range(5)
    }
    reference_ledger = action_key_ledgers["toy-policy-0"]
    assert len(reference_ledger) > cli_module.TOY_HORIZON
    assert all(
        action_key_ledgers[candidate_id] == reference_ledger
        for candidate_id in action_key_ledgers
    )
    fqe_cost = result["estimates"]["FH_FQE_G099_H5"]["toy-policy-0"]["cost"]
    assert fqe_cost["iterations"] > 0
    assert fqe_cost["linear_solves"] == 1
    kmi_cost = result["estimates"]["FH_KMIFQE_G099_H5"]["toy-policy-0"]["cost"]
    assert kmi_cost["hessian_updates"] > 0
    assert kmi_cost["target_updates"] > 0
    etm_cost = result["estimates"]["ETM_MBOPE_G099_H5"]["toy-policy-0"]["cost"]
    assert etm_cost["inference_langevin_gradient_evaluations"] > 0
    assert (output / result["artifacts"]["metrics_json"]).is_file()
    assert (output / result["artifacts"]["metrics_csv"]).is_file()
    assert (output / result["artifacts"]["runtime"]).is_file()
    assert result["artifacts"]["runtime_sha256"] == sha256_file(
        output / result["artifacts"]["runtime"]
    )
    assert (
        result["method_scope"]["FH_KMIFQE_G099_H5"]["production_status"]
        == "NO_GO_OPS_DS_DENSE_HESSIAN_PANEL"
    )
    assert (
        result["method_scope"]["ETM_MBOPE_G099_H5"]["production_status"]
        == "NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT"
    )
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


def test_same_seed_is_candidate_order_invariant_without_float_hash_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    candidates = cli_module._toy_candidates()
    first = run_toy(first_output, seed=23, implementation_commit="d" * 40)
    monkeypatch.setattr(
        cli_module,
        "_toy_candidates",
        lambda: list(reversed(candidates)),
    )
    second = run_toy(second_output, seed=23, implementation_commit="d" * 40)

    # This digest binds only discrete identity/status/ranking semantics.  Float
    # values are compared below with the declared local float64 tolerance.
    assert first["reproducibility_sha256"] == second["reproducibility_sha256"]
    for method_id, first_rows in first["estimates"].items():
        second_rows = second["estimates"][method_id]
        assert set(first_rows) == set(second_rows)
        for candidate_id, first_row in first_rows.items():
            second_row = second_rows[candidate_id]
            assert first_row["method_id"] == second_row["method_id"]
            assert first_row["status"] == second_row["status"]
            assert first_row["value"] == pytest.approx(
                second_row["value"], rel=1e-8, abs=1e-10
            )
    for candidate_id, first_score in first["raw_scores"].items():
        assert first_score == pytest.approx(
            second["raw_scores"][candidate_id], rel=1e-12, abs=1e-14
        )
    for method_id in EXPECTED_METHODS:
        first_ref = first["ranking_seals"][method_id]
        second_ref = second["ranking_seals"][method_id]
        first_seal = load_ranking_seal(
            first_output / first_ref["path"],
            expected_seal_digest=first_ref["sha256"],
        )
        second_seal = load_ranking_seal(
            second_output / second_ref["path"],
            expected_seal_digest=second_ref["sha256"],
        )
        assert first_seal.payload["candidate_set_digest"] == second_seal.payload[
            "candidate_set_digest"
        ]
        assert first_seal.payload["ranking"] == second_seal.payload["ranking"]
        assert first_seal.payload["selected_candidate_id"] == second_seal.payload[
            "selected_candidate_id"
        ]


def test_common_random_key_panel_binds_context_and_excludes_candidate() -> None:
    common = dict(
        seed=41,
        method_id="FH_FQE_G099_H1000",
        phase="estimate_s0_rows",
        rows=4,
    )
    first = cli_module._common_random_keys(context_id="context-a", **common)
    replay = cli_module._common_random_keys(context_id="context-a", **common)
    other_context = cli_module._common_random_keys(context_id="context-b", **common)

    np.testing.assert_array_equal(first, replay)
    assert first.dtype == np.uint64
    assert not np.array_equal(first, other_context)


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


def test_real_smoke_cli_emits_data_gate_as_stable_json(monkeypatch, capsys) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        raise DataValidationError("INVALID_DATA", "actor authority differs")

    monkeypatch.setattr(cli_module, "run_real_smoke", fail)
    assert main(
        [
            "real-smoke",
            "--config",
            "config.json",
            "--expected-config-sha256",
            "a" * 64,
            "--output",
            "output",
        ]
    ) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "INVALID_DATA",
        "detail": "actor authority differs",
    }


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
    assert set(report["method_blockers"]) == {
        "FH_KMIFQE_G099_H1000",
        "ETM_MBOPE_G099_H1000",
    }
    blocker_codes = {
        blocker["code"]
        for blockers in report["method_blockers"].values()
        for blocker in blockers
    }
    assert blocker_codes == {
        "NO_GO_OPS_DS_DENSE_HESSIAN_PANEL",
        "NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT",
    }
    assert {
        blocker["status"]
        for blockers in report["method_blockers"].values()
        for blocker in blockers
    } == {"NO_GO"}
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
    identity_keys = set(report["implementation"])
    required_identity = {"commit", "tree", "worktree_status"}
    assert required_identity <= identity_keys
    if cli_module._verified_source_checkout() is not None:
        assert identity_keys == required_identity
    else:
        assert {"package_name", "package_version"} <= identity_keys
        assert report["implementation"]["worktree_status"] in {
            "INSTALLED_IMMUTABLE_CONTENT",
            "UNVERIFIED_PACKAGE_LAYOUT",
        }
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

    bare_repo = tmp_path / "foreign-consumer.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare_repo)], check=True)
    subprocess.run(
        ["git", f"--git-dir={bare_repo}", "config", "core.bare", "yes"],
        check=True,
    )
    with pytest.raises(PermissionError, match="different Git repository"):
        cli_module._guard_output_location(bare_repo / "artifacts" / "toy")


def test_installed_layout_never_inherits_or_writes_foreign_git_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_root = cli_module._verified_source_checkout()
    if source_root is not None:
        assert source_root == Path(cli_module.__file__).resolve().parents[2]
        source_output = source_root / "artifacts" / "r0-source-layout-check"
        assert cli_module._guard_output_location(source_output) == source_output

    foreign = tmp_path / "foreign-consumer"
    foreign.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=foreign, check=True)
    (foreign / "pyproject.toml").write_text(
        '[project]\nname = "foreign-consumer"\nversion = "1.0"\n',
        encoding="utf-8",
    )
    (foreign / "consumer.txt").write_text("foreign identity\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=foreign, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=R0 Test",
            "-c",
            "user.email=r0@example.invalid",
            "commit",
            "-qm",
            "foreign fixture",
        ],
        cwd=foreign,
        check=True,
    )
    foreign_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=foreign,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    installed_package = foreign / "site" / "policy_learnware_ope"
    shutil.copytree(
        Path(cli_module.__file__).resolve().parent,
        installed_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    monkeypatch.setattr(cli_module, "__file__", str(installed_package / "cli.py"))
    monkeypatch.setattr(cli_module, "distribution_version", lambda _name: "0.4.1b0")

    identity = cli_module._implementation_identity()
    assert identity["package_version"] == "0.4.1b0"
    assert identity["worktree_status"] == "INSTALLED_IMMUTABLE_CONTENT"
    assert identity["tree"] != foreign_head
    assert foreign_head not in identity["commit"]
    with pytest.raises(PermissionError, match="different Git repository"):
        cli_module._guard_output_location(foreign / "artifacts" / "installed-toy")
    outside = tmp_path / "outside-consumer" / "installed-toy"
    assert cli_module._guard_output_location(outside) == outside.resolve()
    assert cli_module._implementation_identity("e" * 40) == {
        "commit": "e" * 40,
        "tree": "CALLER_SUPPLIED",
        "worktree_status": "CALLER_SUPPLIED",
    }


def test_real_smoke_seals_three_methods_with_crn_and_resumes_without_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context_id = "v02q-69d3872f3de8ed010aeca273989c36c1"
    task_id = "CheetahRun"
    candidate_ids = [f"candidate-{index}" for index in range(5)]
    fit_membership = "f" * 64
    census = {
        "schema": "policy-learnware.ope.p0-live-census.v1",
        "freeze": {
            "candidate_sets": {
                task_id: {
                    "candidate_ids": candidate_ids,
                    "membership_digest": "m" * 64,
                    "records": [],
                }
            },
            "memberships": {
                context_id: {
                    "task_id": task_id,
                    "fit_membership_digest": fit_membership,
                    "validation_membership_digest": "v" * 64,
                    "s0_membership_digest": "s" * 64,
                }
            },
            "smoke": {
                "budget_episodes": 24,
                "seed": 1,
                "status": "NO_GO_PRE_ORACLE_SMOKE_BRIDGES_MISSING",
                "contexts": {
                    task_id: {
                        "candidate_membership_digest": "m" * 64,
                        "context_id": context_id,
                        "dataset_digest": "e" * 64,
                        "fit_membership_digest": fit_membership,
                        "validation_membership_digest": "v" * 64,
                        "s0_membership_digest": "s" * 64,
                        "methods": [
                            "RAW_DELTA_TASK5",
                            "FH_FQE_G099_H1000",
                            "DOPE_STYLE_MB_FF_G099_H1000",
                        ],
                    }
                },
            },
        },
        "asset_facts": {
            "banks": {
                "full_rows": [
                    {
                        "context_id": context_id,
                        "task_id": task_id,
                        "bank_sha256": "b" * 64,
                        "dataset_digest": "e" * 64,
                        "role": "development_query",
                        "status": "PASS",
                    }
                ]
            }
        },
    }
    census_path = tmp_path / "p0.json"
    census_path.write_bytes(cli_module._canonical_bytes(census))
    calls: dict[str, object] = {
        "raw": 0,
        "fqe_fit_count": 0,
        "fqe_fit": {},
        "fqe_estimate": {},
        "mb_fit_count": 0,
        "mb_estimate": {},
        "actor_verify": 0,
    }

    def fake_export(*args, output_dir, **kwargs):
        del args, kwargs
        output_dir.mkdir()
        split_digests = {}
        for split in ("fit", "validation", "s0"):
            split_path = output_dir / f"{split}.npz"
            split_path.write_bytes(f"{split}-transitions".encode())
            split_digests[split] = sha256_file(split_path)
        manifest = {
            "context": {"context_id": context_id, "task_id": task_id},
            "source": {"bank_sha256": "b" * 64},
            "membership_protocol": {
                "fit_episode_count": 24,
                "fit_rows_per_episode": 64,
                "split_seed": 40401,
            },
            "splits": {
                "fit": {
                    "membership_digest": fit_membership,
                    "file_sha256": split_digests["fit"],
                },
                "validation": {
                    "membership_digest": "v" * 64,
                    "file_sha256": split_digests["validation"],
                },
                "s0": {
                    "membership_digest": "s" * 64,
                    "file_sha256": split_digests["s0"],
                },
            },
        }
        cli_module._write_canonical_json(output_dir / "manifest.json", manifest)
        return {
            "manifest": manifest,
            "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        }

    def fake_query(*args, output_path, **kwargs):
        del args, kwargs
        output_path.write_bytes(b"six-field-reward-free-query")
        return {
            "artifact_sha256": sha256_file(output_path),
            "fields": [
                "action",
                "episode_offsets",
                "membership_digest",
                "native_timestep",
                "next_observation",
                "observation",
            ],
            "membership_digest": fit_membership,
            "transition_count": 1536,
        }

    batch = _toy_batch(3, episodes=1)
    monkeypatch.setattr(cli_module, "export_existing_log", fake_export)
    monkeypatch.setattr(cli_module, "export_reward_free_query", fake_query)
    monkeypatch.setattr(cli_module, "load_export", lambda *args, **kwargs: batch)
    implementation_identity = {
        "commit": "c" * 40,
        "tree": "d" * 40,
        "worktree_status": "CLEAN",
    }
    monkeypatch.setattr(
        cli_module, "_implementation_identity", lambda: dict(implementation_identity)
    )
    monkeypatch.setattr(
        cli_module,
        "join_oracle_and_score",
        lambda *args, **kwargs: pytest.fail("real smoke must not join labels"),
    )

    class FakeAuthorityFactory:
        @classmethod
        def from_json(cls, *args, candidate_id, expected_sha256, **kwargs):
            del cls, args, kwargs
            return SimpleNamespace(
                candidate_id=candidate_id,
                task_id=task_id,
                authority_sha256=expected_sha256,
                bundle_digest=(candidate_id[-1] * 64),
            )

    class FakeActor:
        semantics = PolicySemantics.STOCHASTIC_KEYED

        def __init__(self, authority, **kwargs):
            del kwargs
            self.authority = authority
            self.policy_id = authority.candidate_id
            self.parity = {
                "status": "PASS",
                "same_key_replay": {"status": "PASS"},
                "different_key_sensitivity": {"status": "PASS"},
            }

        def verify_unchanged(self):
            calls["actor_verify"] = int(calls["actor_verify"]) + 1
            return {"status": "PASS", "candidate_id": self.policy_id}

    class FakeFQE:
        def __init__(self, **kwargs):
            del kwargs
            self.method_id = "FH_FQE_G099_H1000"
            self.actor = None

        def fit(self, batch_value, actor, *, fit_keys):
            assert batch_value is batch
            self.actor = actor
            calls["fqe_fit_count"] = int(calls["fqe_fit_count"]) + 1
            calls["fqe_fit"][actor.policy_id] = np.asarray(fit_keys).tobytes()
            return self

        def estimate(self, initial, *, keys, initial_timestep):
            assert len(initial) == len(initial_timestep)
            calls["fqe_estimate"][self.actor.policy_id] = np.asarray(keys).tobytes()
            score = float(int(self.actor.policy_id[-1]))
            return ValueEstimate(
                method_id=self.method_id,
                status=EstimateStatus.PASS,
                value=score,
                support={"actor_semantics": "stochastic_keyed"},
                provenance={"candidate_id": self.actor.policy_id},
                cost={"fit_transitions": len(batch)},
                diagnostics={"finite": True},
            )

    class FakeMB:
        method_id = "DOPE_STYLE_MB_FF_G099_H1000"

        def fit(self, batch_value, actor, *, fit_keys):
            del actor, fit_keys
            assert batch_value is batch
            calls["mb_fit_count"] = int(calls["mb_fit_count"]) + 1
            return self

        def estimate(self, initial, *, keys, initial_timestep, candidate):
            assert len(initial) == len(initial_timestep)
            calls["mb_estimate"][candidate.policy_id] = np.asarray(keys).tobytes()
            return ValueEstimate(
                method_id=self.method_id,
                status=EstimateStatus.PASS,
                value=float(int(candidate.policy_id[-1])),
                support={"actor_semantics": "stochastic_keyed"},
                provenance={"candidate_id": candidate.policy_id},
                cost={"fit_transitions": len(batch)},
                diagnostics={"finite": True},
            )

    def fake_raw(arguments):
        calls["raw"] = int(calls["raw"]) + 1
        request = arguments["request"]
        binding = {
            "request_schema": request["schema"],
            "method_id": request["method_id"],
            "task_id": request["task_id"],
            "context_id": request["context_id"],
            "candidate_ids": request["candidate_ids"],
            "query": request["query"],
            "membership_digest": request["membership_digest"],
            "operator_authority_sha256": request["operator_authority_sha256"],
        }
        request_sha = cli_module.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "schema": RAW_RESPONSE_SCHEMA,
            "request_binding": binding,
            "request_sha256": request_sha,
            "operator_authority_sha256": request["operator_authority_sha256"],
            "scores": {
                candidate_id: float(index)
                for index, candidate_id in enumerate(request["candidate_ids"])
            },
            "score_semantics": RAW_SCORE_SEMANTICS,
            "synthetic_fixture_only": False,
        }

    monkeypatch.setattr(cli_module, "ActorAuthority", FakeAuthorityFactory)
    monkeypatch.setattr(cli_module, "FrozenFPOActor", FakeActor)
    monkeypatch.setattr(cli_module, "FiniteHorizonFQE", FakeFQE)
    monkeypatch.setattr(cli_module, "make_model_based_estimator", lambda *a, **k: FakeMB())
    monkeypatch.setattr(cli_module, "_execute_raw_isolated", fake_raw)

    def config_value(order):
        return {
            "schema": REAL_SMOKE_CONFIG_SCHEMA,
            "protocol": {
                "context_id": context_id,
                "task_id": task_id,
                "seed": 1,
                "gamma": 0.99,
                "horizon": 1000,
                "budget": 24,
                "split_seed": 40401,
                "track": "development",
            },
            "dataset": {
                "p0_census_path": str(census_path),
                "p0_census_sha256": sha256_file(census_path),
                "bank_path": str(tmp_path / "bank.npz"),
            },
            "actors": {
                "fpo_checkout": str(tmp_path / "fpo-v03"),
                "policy_repo_checkout": str(tmp_path / "actor-v03"),
                "candidates": {
                    candidate_id: {
                        "authority_path": str(tmp_path / f"{candidate_id}.json"),
                        "authority_sha256": str(index) * 64,
                        "bundle_dir": str(tmp_path / candidate_id),
                    }
                    for index, candidate_id in order
                },
            },
            "raw": {
                "authority_path": str(tmp_path / "raw-authority.json"),
                "authority_sha256": "a" * 64,
                "repo_root": str(tmp_path / "raw-v04a"),
                "raw_view_root": str(tmp_path / "raw-view"),
                "asset_census_path": str(tmp_path / "raw-census.json"),
                "raw_adapter_path": str(tmp_path / "raw-adapter.json"),
                "block_size": 2048,
            },
            "fqe": {
                "ridge": 1e-6,
                "max_iterations": 20,
                "tolerance": 1e-3,
                "stochastic_action_samples": 4,
            },
            "mbff": {
                "ridge": 1e-4,
                "rollouts_per_initial": 2,
                "ensemble_members": 2,
                "hidden_dim": 8,
                "termination_mode": "horizon_only",
            },
        }

    first_config = tmp_path / "config-a.json"
    first_config.write_bytes(
        cli_module._canonical_bytes(config_value(list(enumerate(candidate_ids))))
    )
    first = run_real_smoke(
        first_config,
        tmp_path / "run-a",
        expected_config_sha256=sha256_file(first_config),
    )
    implementation_identity["commit"] = "e" * 40
    with pytest.raises(GateClosed, match="resume config lock differs"):
        run_real_smoke(
            first_config,
            tmp_path / "run-a",
            expected_config_sha256=sha256_file(first_config),
            resume=True,
        )
    implementation_identity["commit"] = "c" * 40
    first_run_path = tmp_path / "run-a" / "run.json"
    original_run_bytes = first_run_path.read_bytes()
    tampered_summary = json.loads(original_run_bytes)
    tampered_summary["status"] = "PASS"
    tampered_summary["rankings"]["FH_FQE_G099_H1000"] = list(
        reversed(tampered_summary["rankings"]["FH_FQE_G099_H1000"])
    )
    first_run_path.write_bytes(cli_module._canonical_bytes(tampered_summary))
    with pytest.raises(GateClosed, match="existing run summary differs"):
        run_real_smoke(
            first_config,
            tmp_path / "run-a",
            expected_config_sha256=sha256_file(first_config),
            resume=True,
        )
    first_run_path.write_bytes(original_run_bytes)
    before_resume = json.loads(json.dumps(calls, default=str))
    resumed = run_real_smoke(
        first_config,
        tmp_path / "run-a",
        expected_config_sha256=sha256_file(first_config),
        resume=True,
    )
    assert resumed == first
    assert json.loads(json.dumps(calls, default=str)) == before_resume

    (tmp_path / "run-a" / "run.json").unlink()
    recovered = run_real_smoke(
        first_config,
        tmp_path / "run-a",
        expected_config_sha256=sha256_file(first_config),
        resume=True,
    )
    assert recovered == first
    assert calls["raw"] == 1
    assert calls["fqe_fit_count"] == 5
    assert calls["mb_fit_count"] == 1

    reverse_order = list(reversed(list(enumerate(candidate_ids))))
    second_config = tmp_path / "config-b.json"
    second_config.write_text(
        json.dumps(config_value(reverse_order), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    second = run_real_smoke(
        second_config,
        tmp_path / "run-b",
        expected_config_sha256=sha256_file(second_config),
    )

    assert first["status"] == second["status"] == "SEALED_PRE_ORACLE"
    assert first["metrics_status"] == second["metrics_status"] == "WAITING_ORACLE"
    assert first["oracle_accessed"] is second["oracle_accessed"] is False
    assert set(first["ranking_seals"]) == {
        RAW_PROJECT_METHOD_ID,
        "FH_FQE_G099_H1000",
        "DOPE_STYLE_MB_FF_G099_H1000",
    }
    assert first["rankings"] == second["rankings"]
    assert all(ranking[0] == "candidate-4" for ranking in first["rankings"].values())
    assert len(set(calls["fqe_fit"].values())) == 1
    assert len(set(calls["fqe_estimate"].values())) == 1
    assert len(set(calls["mb_estimate"].values())) == 1
    assert calls["raw"] == 2
    assert calls["fqe_fit_count"] == 10
    assert calls["mb_fit_count"] == 2
    assert calls["actor_verify"] == 15
    for output in (tmp_path / "run-a", tmp_path / "run-b"):
        assert str(tmp_path) not in (output / "run.json").read_text(encoding="utf-8")
        for reference in json.loads(
            (output / "run.json").read_text(encoding="utf-8")
        )["ranking_seals"].values():
            seal_text = (output / reference["path"]).read_text(encoding="utf-8")
            assert str(tmp_path) not in seal_text
            assert "oracle" not in seal_text.casefold()

    invalid = config_value(list(enumerate(candidate_ids)))
    invalid["oracle_path"] = str(tmp_path / "forbidden.json")
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_bytes(cli_module._canonical_bytes(invalid))
    with pytest.raises(GateClosed, match="config fields differ"):
        run_real_smoke(
            invalid_path,
            tmp_path / "invalid-run",
            expected_config_sha256=sha256_file(invalid_path),
        )

    source_census = json.loads(census_path.read_text(encoding="utf-8"))
    source_census["asset_facts"]["banks"]["full_rows"][0]["role"] = "source"
    source_census_path = tmp_path / "source-p0.json"
    source_census_path.write_bytes(cli_module._canonical_bytes(source_census))
    invalid_source = config_value(list(enumerate(candidate_ids)))
    invalid_source["dataset"]["p0_census_path"] = str(source_census_path)
    invalid_source["dataset"]["p0_census_sha256"] = sha256_file(source_census_path)
    invalid_source_path = tmp_path / "source-config.json"
    invalid_source_path.write_bytes(cli_module._canonical_bytes(invalid_source))
    with pytest.raises(GateClosed, match="P0 bank/context binding differs"):
        run_real_smoke(
            invalid_source_path,
            tmp_path / "source-run",
            expected_config_sha256=sha256_file(invalid_source_path),
        )

    estimates_path = tmp_path / "run-b" / "fqe" / "estimates.json"
    estimates_path.write_bytes(estimates_path.read_bytes() + b"tamper")
    with pytest.raises(GateClosed, match="fqe stage artifact mismatch"):
        run_real_smoke(
            second_config,
            tmp_path / "run-b",
            expected_config_sha256=sha256_file(second_config),
            resume=True,
        )

    fit_path = tmp_path / "run-a" / "data" / "transitions" / "fit.npz"
    fit_path.write_bytes(fit_path.read_bytes() + b"tamper")
    with pytest.raises(GateClosed, match="data stage artifact mismatch"):
        run_real_smoke(
            first_config,
            tmp_path / "run-a",
            expected_config_sha256=sha256_file(first_config),
            resume=True,
        )
