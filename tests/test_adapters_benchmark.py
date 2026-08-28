from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from policy_learnware_ope.adapters import (
    ActorAuthority,
    FrozenRepoAuthority,
    GateClosed,
    InProcessActorProvider,
    RawDeltaTask5Adapter,
    SubprocessActorProvider,
    census_real_assets,
    sha256_file,
)
from policy_learnware_ope.benchmark import (
    export_metrics,
    join_oracle_and_score,
    load_ranking_seal,
    seal_ranking,
)


class ToyRawDelegate:
    def __init__(self) -> None:
        self.requests = []

    def scores(self, request):
        self.requests.append(request)
        return {candidate: float(index) for index, candidate in enumerate(request["candidate_ids"])}


def _authority(candidate_id="candidate-0", semantics="stochastic_keyed"):
    return ActorAuthority(
        candidate_id=candidate_id,
        candidate_digest="a" * 64,
        task_id="TASK",
        observation_dim=2,
        action_dim=1,
        observation_abi="obs-v1",
        action_abi="act-v1",
        policy_semantics=semantics,
        normalizer_digest="e" * 64,
        action_scaling_digest="f" * 64,
        repo_commit="b" * 40,
        repo_tree_digest="1" * 40,
        upstream_runtime_commit="c" * 40,
        source_digest="d" * 64,
        dependency_lock_digest="2" * 64,
    )


def test_keyed_stochastic_actor_is_reproducible_and_fails_deterministic_gate():
    authority = _authority()

    def action_fn(candidate_id, observations, native_timestep, keys):
        del candidate_id, native_timestep
        return np.asarray(
            [[np.random.default_rng(int(key)).normal() + row[0]] for row, key in zip(observations, keys)]
        )

    provider = InProcessActorProvider({authority.candidate_id: authority}, action_fn)
    observations = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    keys = np.asarray([10, 20], dtype=np.uint64)
    times = np.asarray([0, 1], dtype=np.int64)
    first = provider.actions(
        authority.candidate_id,
        observations,
        native_timestep=times,
        action_keys=keys,
    )
    second = provider.actions(
        authority.candidate_id,
        observations,
        native_timestep=times,
        action_keys=keys,
    )
    np.testing.assert_array_equal(first, second)
    with pytest.raises(GateClosed, match="NO_GO_TARGET_POLICY_SEMANTICS"):
        provider.actions(
            authority.candidate_id,
            observations,
            native_timestep=times,
            action_keys=None,
        )
    with pytest.raises(GateClosed, match="NO_GO_TARGET_POLICY_SEMANTICS"):
        provider.actions(
            authority.candidate_id,
            observations,
            native_timestep=times,
            action_keys=keys,
            require_deterministic=True,
        )
    bound = provider.bind(authority.candidate_id)
    np.testing.assert_array_equal(
        bound.sample_actions(observations, times, keys=keys),
        first,
    )


def test_raw_task5_delegates_without_reimplementing_operator():
    delegate = ToyRawDelegate()
    adapter = RawDeltaTask5Adapter(delegate)
    candidate_tasks = {f"candidate-{index}": "TASK" for index in range(5)}
    candidate_tasks["other-task"] = "OTHER"
    scores = adapter.score(
        context_id="ctx",
        task_id="TASK",
        candidate_tasks=candidate_tasks,
        query_artifact="opaque-query.npz",
        membership_digest="e" * 64,
    )
    expected = {candidate: float(index) for index, candidate in enumerate(sorted(scores))}
    assert scores == expected
    request = delegate.requests[0]
    assert request["reward_visible"] is False
    assert request["candidate_actions_visible"] is False
    assert request["candidate_ids"] == sorted(scores)


def test_frozen_repo_authority_detects_commit_and_source_drift(tmp_path: Path):
    repo = tmp_path / "frozen"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    source = repo / "operator.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "operator.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    authority = FrozenRepoAuthority(
        repo,
        commit,
        {"operator.py": sha256_file(source)},
        tree_digest=tree,
    )
    verified = authority.verify()
    assert verified["commit"] == commit
    assert verified["tree_digest"] == tree
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(GateClosed, match="not clean"):
        authority.verify()


def test_digest_locked_subprocess_actor_round_trip(tmp_path: Path):
    repo = tmp_path / "actor-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    service = repo / "actor_service.py"
    service.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "actions = [[row[0] + (int(key) % 7) / 100.0] for row, key in "
        "zip(request['observations'], request['action_keys'])]\n"
        "print(json.dumps({'candidate_id': request['candidate_id'], "
        "'candidate_digest': request['candidate_digest'], "
        "'authority_sha256': request['authority_sha256'], "
        "'action_abi': request['action_abi'], 'actions': actions}))\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "actor_service.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "actor service"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    repo_authority = FrozenRepoAuthority(
        repo,
        commit,
        {"actor_service.py": sha256_file(service)},
        tree_digest=tree,
    )
    authority = ActorAuthority(
        candidate_id="candidate-locked",
        candidate_digest="a" * 64,
        task_id="TASK",
        observation_dim=2,
        action_dim=1,
        observation_abi="obs-v1",
        action_abi="act-v1",
        policy_semantics="stochastic_keyed",
        normalizer_digest="b" * 64,
        action_scaling_digest="c" * 64,
        repo_commit=commit,
        repo_tree_digest=tree,
        upstream_runtime_commit="d" * 40,
        source_digest=sha256_file(service),
        dependency_lock_digest="e" * 64,
    )
    provider = SubprocessActorProvider(
        {authority.candidate_id: authority},
        [sys.executable, "actor_service.py"],
        cwd=repo,
        repo_authority=repo_authority,
    ).bind(authority.candidate_id)
    observation = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    timestep = np.asarray([0, 1], dtype=np.int64)
    keys = np.asarray([8, 10], dtype=np.uint64)
    np.testing.assert_allclose(
        provider.sample_actions(observation, timestep, keys=keys),
        [[1.01], [3.03]],
    )


def test_census_rejects_compressed_timestep_unknown_density_and_episodic_oracle(tmp_path: Path):
    dataset = tmp_path / "sampled.npz"
    count = 64
    np.savez(
        dataset,
        observation=np.zeros((count, 2)),
        action=np.zeros((count, 1)),
        reward=np.zeros(count),
        next_observation=np.zeros((count, 2)),
        terminated=np.zeros(count, dtype=bool),
        truncated=np.zeros(count, dtype=bool),
        episode_offsets=np.asarray([0, count]),
        native_timestep=np.arange(count),
    )
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"episode_returns": [1.0, 2.0]}), encoding="utf-8")
    authority = tmp_path / "actor.json"
    _authority(semantics="deterministic").to_json(authority)
    report = census_real_assets(
        dataset_path=dataset,
        oracle_path=oracle,
        actor_authority_path=authority,
    )
    assert report["status"] == "FAIL_CLOSED"
    assert report["native_timestep_status"] == "INVALID_OR_COMPRESSED"
    assert "NO_GO_EXISTING_LOG_DENSITY" in report["gates"]
    assert "NO_GO_ORACLE_DISCOUNTED_VALUE" in report["gates"]


def test_ranking_seal_precedes_oracle_join_and_exports_raw_na(tmp_path: Path):
    seal = seal_ranking(
        tmp_path / "raw.seal.json",
        method_id="RAW_DELTA_TASK5",
        context_id="ctx",
        score_kind="compatibility",
        scores={"a": 0.1, "b": 0.9, "c": 0.2},
        diagnostics={"b": {"runtime_seconds": 0.25}},
        provenance={"membership_digest": "f" * 64},
    )
    assert load_ranking_seal(seal.path).digest == seal.digest
    metrics = join_oracle_and_score(
        seal.path,
        oracle_values={"a": 2.0, "b": 1.0, "c": 3.0},
    )
    assert metrics["hit_at_1"] == 0
    assert metrics["regret_at_1"] == 2.0
    assert metrics["value_mae"] is None
    assert metrics["value_rmse"] is None
    paths = export_metrics(
        [metrics],
        json_path=tmp_path / "metrics.json",
        csv_path=tmp_path / "metrics.csv",
    )
    assert Path(paths["json"]).is_file()
    assert Path(paths["csv"]).is_file()
    assert paths["json_sha256"] == sha256(Path(paths["json"]).read_bytes()).hexdigest()
    csv_text = Path(paths["csv"]).read_text(encoding="utf-8")
    assert "value_mae,value_rmse" in csv_text


def test_value_metrics_are_reported_for_ope_scores(tmp_path: Path):
    seal = seal_ranking(
        tmp_path / "fqe.seal.json",
        method_id="FH_FQE_G099_H1000",
        context_id="ctx",
        score_kind="value",
        scores={"a": 2.5, "b": 1.0, "c": 0.0},
        diagnostics={"a": {"ess": 12.0, "runtime_seconds": 0.1}},
        provenance={"dataset_digest": "a" * 64},
    )
    metrics = join_oracle_and_score(
        seal.path,
        oracle_values={"a": 3.0, "b": 1.0, "c": 1.0},
    )
    assert metrics["hit_at_1"] == 1
    assert metrics["value_mae"] == pytest.approx(0.5)
    assert metrics["value_rmse"] == pytest.approx(np.sqrt(1.25 / 3.0))
    assert metrics["ess_min"] == 12.0


def test_pre_join_seal_rejects_nested_oracle_provenance(tmp_path: Path):
    with pytest.raises(ValueError, match="oracle-bearing"):
        seal_ranking(
            tmp_path / "leaky.json",
            method_id="FH_FQE_G099_H1000",
            context_id="ctx",
            score_kind="value",
            scores={"a": 1.0},
            provenance={"nested": {"oracle_path": "/private/value.json"}},
        )
