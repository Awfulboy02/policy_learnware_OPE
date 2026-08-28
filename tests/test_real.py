from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from policy_learnware_ope.adapters import sha256_file
from policy_learnware_ope.core import DataValidationError, TransitionBatch
import policy_learnware_ope.real as real_module
from policy_learnware_ope.real import (
    ACTOR_AUTHORITY_SCHEMA,
    EXPORT_SCHEMA,
    FROZEN_FPO_PROVIDER_SCHEMA,
    HORIZON,
    P0_CENSUS_SCHEMA,
    SPLIT_SEED,
    ActorAuthority,
    FrozenFPOActor,
    census_bank,
    export_existing_log,
    export_reward_free_query,
    freeze_membership,
    load_export,
)


CONTEXT_ID = "v02q-69d3872f3de8ed010aeca273989c36c1"
EXPECTED_FIT_EPISODES = [
    3, 23, 14, 30, 27, 13, 11, 15, 12, 20, 28, 25,
    0, 18, 5, 24, 1, 29, 6, 2, 22, 21, 19, 10,
]
EXPECTED_FIRST_FIT_TIMESTEPS = [
    0, 3, 25, 40, 56, 57, 58, 87, 89, 97, 104, 117, 135, 141, 163,
    170, 184, 194, 198, 223, 253, 264, 288, 289, 318, 429, 439, 451,
    454, 470, 471, 498, 507, 527, 570, 588, 604, 609, 615, 635, 660,
    663, 680, 720, 726, 729, 741, 802, 812, 815, 817, 831, 843, 853,
    860, 872, 915, 920, 946, 948, 964, 971, 987, 999,
]


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_bank(path: Path, *, mutation: str | None = None) -> None:
    rows = 32 * HORIZON
    row = np.arange(rows, dtype=np.float64)
    observation = np.column_stack((row / rows, np.sin(row / 23.0)))
    action = np.column_stack((row, -row)) / rows
    reward = np.cos(row / 29.0)
    next_observation = observation + np.asarray([0.01, -0.02])
    terminated = np.zeros(rows, dtype=bool)
    truncated = np.zeros(rows, dtype=bool)
    truncated[HORIZON - 1 :: HORIZON] = True
    offsets = np.arange(33, dtype=np.int64) * HORIZON
    if mutation == "nan":
        observation[7, 0] = np.nan
    elif mutation == "shape":
        reward = reward[:-1]
    elif mutation == "offset":
        offsets[4] += 1
    np.savez_compressed(
        path,
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        terminated=terminated,
        truncated=truncated,
        episode_offsets=offsets,
    )


def _write_census(
    path: Path,
    bank: Path,
    *,
    candidate_id: str | None = None,
    candidate_record: dict | None = None,
) -> str:
    bank_sha = sha256_file(bank)
    membership = freeze_membership(CONTEXT_ID, bank_sha)
    frozen = {
        "fit_episode_ids": membership["fit"]["episode_ids"],
        "fit_membership_digest": membership["fit"]["membership_digest"],
        "fit_transition_count": 24 * 64,
        "validation_episode_ids": membership["validation"]["episode_ids"],
        "validation_membership_digest": membership["validation"]["membership_digest"],
        "validation_transition_count": 4 * HORIZON,
        "s0_episode_ids": membership["s0"]["episode_ids"],
        "s0_membership_digest": membership["s0"]["membership_digest"],
        "s0_state_count": 4,
        "task_id": "CheetahRun",
    }
    freeze = {
        "memberships": {CONTEXT_ID: frozen},
        "physical_membership_protocol": {
            "version": "ope-existing-log-membership-v1"
        },
        "split_seed": SPLIT_SEED,
    }
    if candidate_id is not None and candidate_record is not None:
        records = [candidate_record]
        freeze["candidate_sets"] = {
            "CheetahRun": {
                "candidate_ids": [candidate_id],
                "membership_digest": hashlib.sha256(
                    _canonical_bytes(records)
                ).hexdigest(),
                "records": records,
            }
        }
    census = {
        "asset_facts": {
            "banks": {
                "full_rows": [
                    {
                        "action_dim": 2,
                        "bank_sha256": bank_sha,
                        "context_id": CONTEXT_ID,
                        "dataset_digest": "d" * 64,
                        "episode_count": 32,
                        "observation_dim": 2,
                        "role": "development_query",
                        "status": "PASS",
                        "task_id": "CheetahRun",
                        "transition_count": 32 * HORIZON,
                    }
                ]
            }
        },
        "freeze": freeze,
        "schema": P0_CENSUS_SCHEMA,
        "status": "NO_GO",
    }
    path.write_bytes(_canonical_bytes(census))
    return sha256_file(path)


def _source_identity(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return sha256_file(path), stat.st_mtime_ns, stat.st_size


def test_r2_membership_protocol_matches_frozen_p0_fixture() -> None:
    membership = freeze_membership(CONTEXT_ID, "a" * 64)

    assert membership["fit"]["episode_ids"] == EXPECTED_FIT_EPISODES
    assert [
        row["native_timestep"] for row in membership["fit"]["rows"][:64]
    ] == EXPECTED_FIRST_FIT_TIMESTEPS
    assert membership["validation"]["episode_ids"] == [26, 8, 31, 16]
    assert membership["s0"]["episode_ids"] == [17, 9, 4, 7]


def test_export_is_no_clobber_digest_locked_and_loads_strict_batches(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.npz"
    census = tmp_path / "p0.json"
    _write_bank(bank)
    census_sha = _write_census(census, bank)
    source_before = _source_identity(bank)

    summary = census_bank(
        census,
        expected_census_sha256=census_sha,
        context_id=CONTEXT_ID,
        bank_path=bank,
    )
    output = tmp_path / "export"
    result = export_existing_log(
        census,
        expected_census_sha256=census_sha,
        context_id=CONTEXT_ID,
        bank_path=bank,
        output_dir=output,
    )

    assert summary["status"] == "PASS"
    assert summary["source_hash_unchanged"] is True
    assert summary["source_mtime_unchanged"] is True
    assert _source_identity(bank) == source_before
    assert result["manifest"]["schema"] == EXPORT_SCHEMA
    assert result["manifest"]["source"]["read_only_projection"] is True
    assert result["manifest_sha256"] == sha256_file(output / "manifest.json")

    fit = load_export(
        output, "fit", expected_manifest_sha256=result["manifest_sha256"]
    )
    validation = load_export(
        output, "validation", expected_manifest_sha256=result["manifest_sha256"]
    )
    s0 = load_export(
        output, "s0", expected_manifest_sha256=result["manifest_sha256"]
    )
    assert isinstance(fit, TransitionBatch)
    assert (len(fit), fit.episode_count) == (24 * 64, 24)
    assert (len(validation), validation.episode_count) == (4 * HORIZON, 4)
    assert (len(s0), s0.episode_count) == (4, 4)
    assert fit.timestep_provenance == validation.timestep_provenance == "native_indices"
    assert np.array_equal(fit.native_timestep[:64], EXPECTED_FIRST_FIT_TIMESTEPS)
    assert not np.any(fit.dataset_cut)
    assert not np.any(validation.dataset_cut)
    assert np.all(s0.dataset_cut)
    assert np.all(s0.native_timestep == 0)

    with np.load(output / "fit.npz", allow_pickle=False) as exported, np.load(
        bank, allow_pickle=False
    ) as source:
        rows = exported["source_row_index"]
        valid = exported["next_behavior_action_valid"]
        assert np.array_equal(exported["episode_id"], rows // HORIZON)
        assert np.array_equal(exported["native_timestep"], rows % HORIZON)
        assert np.array_equal(exported["terminated"], source["terminated"][rows])
        assert np.array_equal(exported["truncated"], source["truncated"][rows])
        assert np.array_equal(
            exported["next_behavior_action"][valid], source["action"][rows[valid] + 1]
        )
        assert np.all(exported["native_timestep"][~valid] == HORIZON - 1)

    before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    with pytest.raises(FileExistsError, match="no-clobber"):
        export_existing_log(
            census,
            expected_census_sha256=census_sha,
            context_id=CONTEXT_ID,
            bank_path=bank,
            output_dir=output,
        )
    assert before == {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }


def test_reward_free_query_is_exact_digest_locked_and_replay_stable(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.npz"
    census = tmp_path / "p0.json"
    _write_bank(bank)
    census_sha = _write_census(census, bank)
    exported = export_existing_log(
        census,
        expected_census_sha256=census_sha,
        context_id=CONTEXT_ID,
        bank_path=bank,
        output_dir=tmp_path / "export",
    )

    first = tmp_path / "query-1.npz"
    second = tmp_path / "query-2.npz"
    first_summary = export_reward_free_query(
        tmp_path / "export",
        expected_manifest_sha256=exported["manifest_sha256"],
        output_path=first,
    )
    second_summary = export_reward_free_query(
        tmp_path / "export",
        expected_manifest_sha256=exported["manifest_sha256"],
        output_path=second,
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_summary == second_summary
    assert first_summary["fields"] == [
        "action",
        "episode_offsets",
        "membership_digest",
        "native_timestep",
        "next_observation",
        "observation",
    ]
    with np.load(first, allow_pickle=False) as query:
        assert set(query.files) == set(first_summary["fields"])
        assert all(
            forbidden not in query.files
            for forbidden in ("reward", "terminated", "truncated", "oracle")
        )
        assert np.all(np.diff(query["episode_offsets"]) == 64)
        assert str(query["membership_digest"].item()) == first_summary[
            "membership_digest"
        ]
    with pytest.raises(FileExistsError, match="no-clobber"):
        export_reward_free_query(
            tmp_path / "export",
            expected_manifest_sha256=exported["manifest_sha256"],
            output_path=first,
        )


@pytest.mark.parametrize("mutation", ["nan", "shape", "offset"])
def test_invalid_bank_never_leaves_partial_export(
    tmp_path: Path, mutation: str
) -> None:
    bank = tmp_path / f"{mutation}.npz"
    census = tmp_path / f"{mutation}.json"
    _write_bank(bank, mutation=mutation)
    census_sha = _write_census(census, bank)
    before = _source_identity(bank)
    output = tmp_path / f"out-{mutation}"

    with pytest.raises(DataValidationError):
        export_existing_log(
            census,
            expected_census_sha256=census_sha,
            context_id=CONTEXT_ID,
            bank_path=bank,
            output_dir=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))
    assert _source_identity(bank) == before


def test_digest_and_manifest_tampering_fail_closed_without_partial_output(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.npz"
    census = tmp_path / "p0.json"
    _write_bank(bank)
    census_sha = _write_census(census, bank)

    with pytest.raises(DataValidationError, match="census digest mismatch"):
        export_existing_log(
            census,
            expected_census_sha256="0" * 64,
            context_id=CONTEXT_ID,
            bank_path=bank,
            output_dir=tmp_path / "wrong-census",
        )
    assert not (tmp_path / "wrong-census").exists()

    changed_bank = tmp_path / "changed-bank.npz"
    changed_census = tmp_path / "changed-p0.json"
    _write_bank(changed_bank)
    changed_census_sha = _write_census(changed_census, changed_bank)
    with np.load(changed_bank, allow_pickle=False) as archive:
        changed = {name: np.array(archive[name], copy=True) for name in archive.files}
    changed["reward"][0] += 1.0
    np.savez_compressed(changed_bank, **changed)
    changed_identity = _source_identity(changed_bank)
    with pytest.raises(DataValidationError, match="bank digest disagrees"):
        export_existing_log(
            changed_census,
            expected_census_sha256=changed_census_sha,
            context_id=CONTEXT_ID,
            bank_path=changed_bank,
            output_dir=tmp_path / "wrong-bank",
        )
    assert not (tmp_path / "wrong-bank").exists()
    assert _source_identity(changed_bank) == changed_identity

    output = tmp_path / "good"
    result = export_existing_log(
        census,
        expected_census_sha256=census_sha,
        context_id=CONTEXT_ID,
        bank_path=bank,
        output_dir=output,
    )
    with pytest.raises(DataValidationError, match="manifest digest mismatch"):
        load_export(output, "fit", expected_manifest_sha256="f" * 64)
    fit_path = output / "fit.npz"
    os.chmod(fit_path, 0o600)
    fit_path.write_bytes(fit_path.read_bytes() + b"tamper")
    with pytest.raises(DataValidationError, match="split file digest mismatch"):
        load_export(
            output, "fit", expected_manifest_sha256=result["manifest_sha256"]
        )


def _git_repo(path: Path, filename: str) -> tuple[str, str]:
    path.mkdir()
    (path / filename).parent.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text("frozen source\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=OPE Test",
            "-c",
            "user.email=ope@example.invalid",
            "commit",
            "-q",
            "-m",
            "freeze",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    return commit, tree


def _write_actor_bundle(
    root: Path,
    *,
    fpo_commit: str,
    runtime_digest: str,
) -> str:
    root.mkdir()
    np.savez(root / "actor.npz", layer=np.asarray([1.0], dtype=np.float32))
    golden_observation = np.zeros((8, 2), dtype=np.float32)
    golden_raw = np.full((8, 2), np.float32(0.001), dtype=np.float32)
    np.savez(
        root / "golden_io.npz",
        observation=golden_observation,
        prng_key_data=np.asarray([0, 1], dtype=np.uint32),
        raw_action=golden_raw,
        environment_action=np.tanh(golden_raw).astype(np.float32),
    )
    np.savez(
        root / "obs_stats.npz",
        count=np.asarray(1.0),
        mean=np.zeros(2),
        var_sum=np.ones(2),
        std=np.ones(2),
    )
    (root / "policy_spec.json").write_bytes(
        _canonical_bytes(
            {
                "observation_size": 2,
                "action_size": 2,
                "training_config": {},
            }
        )
    )
    (root / "provenance.json").write_bytes(
        _canonical_bytes(
            {
                "fpo_commit": fpo_commit,
                "runtime_digest": runtime_digest,
            }
        )
    )
    filenames = {
        "actor.npz",
        "golden_io.npz",
        "obs_stats.npz",
        "policy_spec.json",
        "provenance.json",
    }
    files = {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        for name in sorted(filenames)
    }
    manifest = {
        "schema": "policy-learnware.policy-bundle.v0",
        "complete": True,
        "created_at": "synthetic",
        "algorithm": "fpo",
        "task": "CheetahRun",
        "seed": 0,
        "outer_iteration": 183,
        "environment_steps": 1,
        "files": files,
    }
    (root / "bundle_manifest.json").write_bytes(_canonical_bytes(manifest))
    return sha256_file(root / "bundle_manifest.json")


class _FakeRandom:
    @staticmethod
    def wrap_key_data(value):
        return np.asarray(value, dtype=np.uint32)

    @staticmethod
    def key_data(value):
        return np.asarray(value, dtype=np.uint32)

    @staticmethod
    def split(key, count):
        assert count == 2
        value = np.asarray(key, dtype=np.uint32)
        return np.stack(
            (
                value ^ np.asarray([0x13579BDF, 0x2468ACE0], dtype=np.uint32),
                value + np.asarray([1, 17], dtype=np.uint32),
            )
        )

    @staticmethod
    def fold_in(key, index):
        value = np.asarray(key, dtype=np.uint32)
        token = np.asarray(index, dtype=np.uint32)
        return value + np.asarray([token, token * np.uint32(17)], dtype=np.uint32)


class _FakeJax:
    random = _FakeRandom()

    def __init__(self):
        self.jit_count = 0

    @staticmethod
    def vmap(function):
        def mapped(*arrays):
            return np.stack(
                [function(*(array[index] for array in arrays)) for index in range(len(arrays[0]))]
            )

        return mapped

    def jit(self, function):
        self.jit_count += 1
        return function

    @staticmethod
    def device_get(value):
        return value


class _FakeJnp:
    float32 = np.float32
    uint32 = np.uint32
    asarray = staticmethod(np.asarray)
    tanh = staticmethod(np.tanh)


class _FakeNativeState:
    def __init__(self, *, bias: float = 0.0):
        self.calls: list[tuple[np.ndarray, np.ndarray, bool]] = []
        self.bias = np.float32(bias)

    def sample_action(self, observation, key, *, deterministic):
        observation = np.asarray(observation, dtype=np.float32)
        key_data = np.asarray(key, dtype=np.uint32)
        self.calls.append((observation.copy(), key_data.copy(), deterministic))
        token = np.float32(key_data[0] * np.float32(0.01) + key_data[1] * np.float32(0.001))
        return observation[..., :2] + token + self.bias, key


def _actor_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fpo = tmp_path / "fpo"
    policy_repo = tmp_path / "policy-repo"
    fpo_commit, fpo_tree = _git_repo(fpo, "playground/src/flow_policy/fpo.py")
    policy_commit, policy_tree = _git_repo(
        policy_repo, "src/policy_learnware_v0/policy/loader.py"
    )
    monkeypatch.setattr(real_module, "FROZEN_FPO_COMMIT", fpo_commit)
    monkeypatch.setattr(real_module, "FROZEN_FPO_TREE", fpo_tree)
    runtime_digest = "9" * 64
    bundle = tmp_path / "bundle"
    bundle_digest = _write_actor_bundle(
        bundle, fpo_commit=fpo_commit, runtime_digest=runtime_digest
    )
    execution_abi = {
        "schema": "policy-learnware.v02-execution-abi.v0",
        "protocol_family_id": "continuous-vector-mdp-v02",
        "policy_runtime_id": "legacy-ppo-fpo-v0",
        "state_abi_id": "stateless-v0",
        "observation_tensor_abi_digest": "1" * 64,
        "action_tensor_abi_digest": "2" * 64,
        "action_transform_id": "tanh",
    }
    candidate_id = "lw-test-frozen-fpo"
    candidate_record = {
        "bundle_digest": bundle_digest,
        "execution_abi": execution_abi,
        "opaque_learnware_id": candidate_id,
        "source_anchor_id": "3" * 64,
        "task_id": "CheetahRun",
    }
    bank = tmp_path / "bank.npz"
    census = tmp_path / "p0.json"
    _write_bank(bank)
    census_sha = _write_census(
        census,
        bank,
        candidate_id=candidate_id,
        candidate_record=candidate_record,
    )
    fpo_source = real_module._checkout_source_identity(fpo)
    authority_payload = {
        "schema": ACTOR_AUTHORITY_SCHEMA,
        "provider_schema": FROZEN_FPO_PROVIDER_SCHEMA,
        "candidate_id": candidate_id,
        "bundle_digest": bundle_digest,
        "bundle_manifest": {
            "schema": "policy-learnware.policy-bundle.v0",
            "sha256": bundle_digest,
        },
        "execution_abi": execution_abi,
        "observation_dim": 2,
        "action_dim": 2,
        "runtime_digest": runtime_digest,
        "fpo_checkout": {
            "commit": fpo_commit,
            "tree": fpo_tree,
            "head_tree_digest": fpo_source["head_tree_digest"],
            "source_digest": fpo_source["source_digest"],
            "source_file_count": fpo_source["source_file_count"],
        },
        "policy_repo_checkout": {
            "commit": policy_commit,
            "tree": policy_tree,
        },
        "loader": {
            "module": "policy_learnware_v0.policy.loader",
            "function": "load_policy",
            "runtime_only": True,
        },
    }
    authority_path = tmp_path / "actor-authority.json"
    authority_path.write_bytes(_canonical_bytes(authority_payload))
    authority_sha = sha256_file(authority_path)
    authority = ActorAuthority.from_json(
        authority_path,
        expected_sha256=authority_sha,
        census_path=census,
        expected_census_sha256=census_sha,
        context_id=CONTEXT_ID,
        candidate_id=candidate_id,
    )
    state = _FakeNativeState()
    loader_calls = []

    def fake_loader(**kwargs):
        loader_calls.append(kwargs)
        return SimpleNamespace(
            native_state=state,
            observation_dim=2,
            action_dim=2,
            bundle_digest=bundle_digest,
        )

    fake_jax = _FakeJax()
    monkeypatch.setattr(real_module, "_load_policy_runtime", fake_loader)
    monkeypatch.setattr(real_module, "_jax_modules", lambda: (fake_jax, _FakeJnp))
    return SimpleNamespace(
        authority=authority,
        authority_payload=authority_payload,
        authority_path=authority_path,
        authority_sha=authority_sha,
        bank=bank,
        bundle=bundle,
        candidate_id=candidate_id,
        census=census,
        census_sha=census_sha,
        execution_abi=execution_abi,
        fake_jax=fake_jax,
        fake_loader=fake_loader,
        fpo=fpo,
        loader_calls=loader_calls,
        policy_repo=policy_repo,
        state=state,
    )


def test_frozen_fpo_actor_uses_only_caller_keys_and_exports_next_key_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _actor_fixture(tmp_path, monkeypatch)
    actor = FrozenFPOActor(
        fixture.authority,
        bundle_dir=fixture.bundle,
        fpo_checkout=fixture.fpo,
        policy_repo_checkout=fixture.policy_repo,
    )
    initial_call_count = len(fixture.state.calls)
    observations = np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    times = np.asarray([7, 19], dtype=np.int64)
    scalar_keys = np.asarray([1, 2**32 + 3], dtype=np.uint64)

    assert np.array_equal(
        actor.next_action_keys(scalar_keys),
        np.asarray([[1, 18], [2, 20]], dtype=np.uint32),
    )
    first = actor.sample_actions(observations, times, keys=scalar_keys)
    second = actor.sample_actions(observations, times, keys=scalar_keys)
    changed = actor.sample_actions(
        observations, times, keys=np.asarray([2, 2**32 + 4], dtype=np.uint64)
    )

    assert actor.policy_id == fixture.candidate_id
    assert actor.semantics.value == "stochastic_keyed"
    assert actor.provenance["historical_deterministic_argument_is_sampling_free"] is False
    assert actor.provenance["initialization_parity"]["status"] == "PASS"
    assert (
        actor.provenance["initialization_parity"]["different_key_sensitivity"][
            "status"
        ]
        == "PASS"
    )
    assert {
        actor.parity[name]["status"]
        for name in (
            "golden_replay",
            "same_key_replay",
            "scalar_compiled",
            "action_transform",
        )
    } == {"PASS"}
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert fixture.fake_jax.jit_count == 1
    assert all(call[2] is True for call in fixture.state.calls)
    assert np.array_equal(fixture.state.calls[initial_call_count][1], [0, 1])
    assert np.array_equal(fixture.state.calls[initial_call_count + 1][1], [1, 3])
    assert fixture.loader_calls[0]["authority"] is fixture.authority
    assert actor.last_key_ledger == {
        "mapping": "scalar_u64->[high32,low32];structured_words=checked_uint32",
        "input_key_data": [[0, 2], [1, 4]],
        "next_key_data": [[1, 19], [2, 21]],
        "transition": "jax.random.split(key,2)[1]",
    }

    structured = np.asarray([[17, 23], [29, 31]], dtype=np.uint64)
    assert np.array_equal(
        actor.next_action_keys(structured),
        np.asarray([[18, 40], [30, 48]], dtype=np.uint32),
    )
    actor.sample_actions(observations, times, keys=structured)
    assert actor.last_key_ledger["input_key_data"] == structured.tolist()
    assert actor.verify_unchanged()["status"] == "PASS"
    with pytest.raises(DataValidationError, match="fit uint32"):
        actor.sample_actions(
            observations,
            times,
            keys=np.asarray([[0, 2**32], [1, 2]], dtype=np.uint64),
        )
    with pytest.raises(DataValidationError, match="non-negative"):
        actor.sample_actions(
            observations,
            times,
            keys=np.asarray([[0, -1], [1, 2]], dtype=np.int64),
        )


def test_frozen_fpo_actor_reports_bounded_cross_backend_golden_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _actor_fixture(tmp_path, monkeypatch)
    fixture.state.bias = np.float32(0.02)
    actor = FrozenFPOActor(
        fixture.authority,
        bundle_dir=fixture.bundle,
        fpo_checkout=fixture.fpo,
        policy_repo_checkout=fixture.policy_repo,
    )

    assert actor.parity["status"] == "WARNING_CROSS_BACKEND_COMPATIBLE"
    golden = actor.parity["golden_replay"]
    assert golden["status"] == "WARNING_CROSS_BACKEND_COMPATIBLE"
    assert golden["raw_max_abs_error"] == pytest.approx(0.02, abs=1e-6)
    assert golden["raw_max_relative_error"] > 0.0
    assert golden["evidence"]["version"] == "v04a-selected-market-f32-m2cpu-v2"
    assert actor.parity["same_key_replay"]["status"] == "PASS"
    assert actor.parity["scalar_compiled"]["status"] == "PASS"
    assert actor.parity["action_transform"]["status"] == "PASS"

    fixture.state.bias = np.float32(0.08)
    with pytest.raises(DataValidationError, match="compatibility envelope"):
        FrozenFPOActor(
            fixture.authority,
            bundle_dir=fixture.bundle,
            fpo_checkout=fixture.fpo,
            policy_repo_checkout=fixture.policy_repo,
        )


def test_actor_authority_and_live_assets_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _actor_fixture(tmp_path, monkeypatch)
    common = {
        "census_path": fixture.census,
        "expected_census_sha256": fixture.census_sha,
        "context_id": CONTEXT_ID,
        "candidate_id": fixture.candidate_id,
    }
    with pytest.raises(DataValidationError, match="authority digest mismatch"):
        ActorAuthority.from_json(
            fixture.authority_path, expected_sha256="0" * 64, **common
        )

    for name, change, message in (
        ("truthy", {"validated": True}, "fields differ"),
        ("oracle", {"oracle_path": "/secret/oracle"}, "environment/oracle"),
    ):
        payload = {**fixture.authority_payload, **change}
        path = tmp_path / f"{name}.json"
        path.write_bytes(_canonical_bytes(payload))
        with pytest.raises(DataValidationError, match=message):
            ActorAuthority.from_json(
                path, expected_sha256=sha256_file(path), **common
            )

    abi_payload = json.loads(json.dumps(fixture.authority_payload))
    abi_payload["execution_abi"]["action_tensor_abi_digest"] = "f" * 64
    abi_path = tmp_path / "abi.json"
    abi_path.write_bytes(_canonical_bytes(abi_payload))
    with pytest.raises(DataValidationError, match="execution ABI differs"):
        ActorAuthority.from_json(
            abi_path, expected_sha256=sha256_file(abi_path), **common
        )

    source_payload = json.loads(json.dumps(fixture.authority_payload))
    source_payload["fpo_checkout"]["source_digest"] = "e" * 64
    source_path = tmp_path / "source.json"
    source_path.write_bytes(_canonical_bytes(source_payload))
    source_authority = ActorAuthority.from_json(
        source_path, expected_sha256=sha256_file(source_path), **common
    )
    with pytest.raises(DataValidationError, match="source digest differs"):
        FrozenFPOActor(
            source_authority,
            bundle_dir=fixture.bundle,
            fpo_checkout=fixture.fpo,
            policy_repo_checkout=fixture.policy_repo,
        )

    def dirty_loader(**kwargs):
        (fixture.policy_repo / "import-side-effect.pyc").write_bytes(b"bytecode")
        return SimpleNamespace(
            native_state=fixture.state,
            observation_dim=2,
            action_dim=2,
            bundle_digest=fixture.authority.bundle_digest,
        )

    monkeypatch.setattr(real_module, "_load_policy_runtime", dirty_loader)
    with pytest.raises(DataValidationError, match="not clean"):
        FrozenFPOActor(
            fixture.authority,
            bundle_dir=fixture.bundle,
            fpo_checkout=fixture.fpo,
            policy_repo_checkout=fixture.policy_repo,
        )
    (fixture.policy_repo / "import-side-effect.pyc").unlink()
    monkeypatch.setattr(real_module, "_load_policy_runtime", fixture.fake_loader)

    (fixture.fpo / "dirty.tmp").write_text("dirty", encoding="utf-8")
    with pytest.raises(DataValidationError, match="not clean"):
        FrozenFPOActor(
            fixture.authority,
            bundle_dir=fixture.bundle,
            fpo_checkout=fixture.fpo,
            policy_repo_checkout=fixture.policy_repo,
        )
    (fixture.fpo / "dirty.tmp").unlink()

    actor_payload = fixture.bundle / "actor.npz"
    actor_payload.write_bytes(actor_payload.read_bytes() + b"tamper")
    with pytest.raises(DataValidationError, match="payload differs"):
        FrozenFPOActor(
            fixture.authority,
            bundle_dir=fixture.bundle,
            fpo_checkout=fixture.fpo,
            policy_repo_checkout=fixture.policy_repo,
        )

    other_fpo = tmp_path / "other-fpo"
    _git_repo(other_fpo, "playground/src/flow_policy/fpo.py")
    with pytest.raises(DataValidationError, match="HEAD differs"):
        FrozenFPOActor(
            fixture.authority,
            bundle_dir=fixture.bundle,
            fpo_checkout=other_fpo,
            policy_repo_checkout=fixture.policy_repo,
        )
