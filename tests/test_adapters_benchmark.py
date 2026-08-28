from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import policy_learnware_ope.adapters as adapters_module
from policy_learnware_ope.adapters import (
    ActorAuthority,
    FrozenRepoAuthority,
    GateClosed,
    InProcessActorProvider,
    RAW_EXECUTION_AUTHORITY_SCHEMA,
    RAW_FIXTURE_METHOD_ID,
    RAW_QUERY_SCHEMA,
    RAW_PROJECT_METHOD_ID,
    RAW_REQUEST_SCHEMA,
    RAW_RESPONSE_SCHEMA,
    RAW_SCORE_SEMANTICS,
    RawDeltaTask5Adapter,
    SealedRawOperator,
    census_real_assets,
    execute_frozen_raw_query,
    sha256_file,
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
    for invalid_keys in (np.asarray([1.5, 2.5]), np.asarray([-1, 2])):
        with pytest.raises(GateClosed, match="integer|non-negative"):
            provider.actions(
                authority.candidate_id,
                observations,
                native_timestep=times,
                action_keys=invalid_keys,
            )


def test_raw_task5_delegates_without_reimplementing_operator():
    delegate = ToyRawDelegate()
    with pytest.raises(GateClosed, match="NO_GO_RAW_OPERATOR_AUTHORITY"):
        RawDeltaTask5Adapter(delegate)
    adapter = RawDeltaTask5Adapter(delegate, method_id="RAW_ADAPTER_FIXTURE")
    candidate_tasks = {f"candidate-{index}": "TASK" for index in range(5)}
    candidate_tasks["other-task"] = "OTHER"
    scores = adapter.score(
        context_id="ctx",
        task_id="TASK",
        candidate_tasks=candidate_tasks,
        query_artifact_digest="d" * 64,
        membership_digest="e" * 64,
    )
    expected = {candidate: float(index) for index, candidate in enumerate(sorted(scores))}
    assert scores == expected
    request = delegate.requests[0]
    assert request["schema"] == RAW_REQUEST_SCHEMA
    assert request["method_id"] == "RAW_ADAPTER_FIXTURE"
    assert request["query"] == {
        "schema": RAW_QUERY_SCHEMA,
        "artifact_sha256": "d" * 64,
        "fields": [
            "observation",
            "action",
            "next_observation",
            "native_timestep",
            "episode_offsets",
        ],
        "forbidden_fields": ["reward", "oracle", "candidate_action"],
    }
    assert request["candidate_ids"] == sorted(scores)

    with pytest.raises(GateClosed, match="reward-free query schema"):
        adapter.score(
            context_id="ctx",
            task_id="TASK",
            candidate_tasks=candidate_tasks,
            query_artifact_digest="d" * 64,
            membership_digest="e" * 64,
            query_schema="ambiguous-query.v0",
        )


@pytest.mark.parametrize("invalid_score", [True, "1.25"])
def test_raw_adapter_rejects_bool_and_numeric_string_scores(invalid_score):
    class InvalidRawDelegate:
        def scores(self, request):
            return {
                candidate: invalid_score if index == 0 else float(index)
                for index, candidate in enumerate(request["candidate_ids"])
            }

    adapter = RawDeltaTask5Adapter(
        InvalidRawDelegate(), method_id="RAW_ADAPTER_FIXTURE"
    )
    with pytest.raises(GateClosed, match="must be a JSON number"):
        adapter.score(
            context_id="ctx",
            task_id="TASK",
            candidate_tasks={f"candidate-{index}": "TASK" for index in range(5)},
            query_artifact_digest="d" * 64,
            membership_digest="e" * 64,
        )


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


def test_production_raw_and_self_promoted_identity_remain_no_go():
    with pytest.raises(GateClosed, match="caller-pinned response and authority"):
        RawDeltaTask5Adapter(ToyRawDelegate(), method_id=RAW_PROJECT_METHOD_ID)
    with pytest.raises(GateClosed, match="unsupported Raw method identity"):
        RawDeltaTask5Adapter(
            ToyRawDelegate(), method_id="RAW_DELTA_TASK5_OFFICIAL_PARITY"
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
        dataset_cut=np.zeros(count, dtype=bool),
        episode_id=np.zeros(count, dtype=np.int64),
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
    assert report["status"] == "NO_GO"
    assert report["native_timestep_status"] == "INVALID_OR_COMPRESSED"
    assert "NO_GO_EXISTING_LOG_DENSITY" in report["gates"]
    assert "NO_GO_ORACLE_DISCOUNTED_VALUE" in report["gates"]


def _raw_request_binding() -> dict:
    return {
        "request_schema": RAW_REQUEST_SCHEMA,
        "method_id": "RAW_ADAPTER_FIXTURE",
        "task_id": "TASK",
        "context_id": "ctx",
        "candidate_ids": [f"candidate-{index}" for index in range(5)],
        "query": {
            "schema": RAW_QUERY_SCHEMA,
            "artifact_sha256": "d" * 64,
            "fields": [
                "observation",
                "action",
                "next_observation",
                "native_timestep",
                "episode_offsets",
            ],
            "forbidden_fields": ["reward", "oracle", "candidate_action"],
        },
        "membership_digest": "e" * 64,
    }


def _write_raw_response(
    path: Path,
    binding: dict,
    *,
    schema: str = RAW_RESPONSE_SCHEMA,
    authority_digest: str | None = None,
) -> None:
    request_digest = sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "request_binding": binding,
                "request_sha256": request_digest,
                "scores": {
                    f"candidate-{index}": float(index) for index in range(5)
                },
                "synthetic_fixture_only": authority_digest is None,
                **(
                    {
                        "operator_authority_sha256": authority_digest,
                        "score_semantics": RAW_SCORE_SEMANTICS,
                    }
                    if authority_digest is not None
                    else {}
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "request_schema",
        "method_id",
        "task_id",
        "context_id",
        "candidate_ids",
        "query_schema",
        "query_artifact_digest",
        "membership_digest",
    ],
)
def test_sealed_raw_response_binds_every_request_identity(tmp_path: Path, mutation: str):
    binding = _raw_request_binding()
    if mutation == "candidate_ids":
        binding["candidate_ids"] = ["intruder", *binding["candidate_ids"][1:]]
    elif mutation == "query_schema":
        binding["query"]["schema"] = "wrong-query-schema"
    elif mutation == "query_artifact_digest":
        binding["query"]["artifact_sha256"] = "0" * 64
    else:
        binding[mutation] = "wrong"
    artifact = tmp_path / f"{mutation}.json"
    _write_raw_response(artifact, binding)
    adapter = RawDeltaTask5Adapter(
        SealedRawOperator(artifact, sha256_file(artifact)),
        method_id="RAW_ADAPTER_FIXTURE",
    )
    with pytest.raises(GateClosed, match="request binding mismatch"):
        adapter.score(
            context_id="ctx",
            task_id="TASK",
            candidate_tasks={f"candidate-{index}": "TASK" for index in range(5)},
            query_artifact_digest="d" * 64,
            membership_digest="e" * 64,
        )


def test_sealed_raw_response_schema_and_digest_fail_closed(tmp_path: Path):
    binding = _raw_request_binding()
    artifact = tmp_path / "response.json"
    _write_raw_response(artifact, binding, schema="raw-response-self-asserted.v0")
    adapter = RawDeltaTask5Adapter(
        SealedRawOperator(artifact, sha256_file(artifact)),
        method_id="RAW_ADAPTER_FIXTURE",
    )
    with pytest.raises(GateClosed, match="response schema mismatch"):
        adapter.score(
            context_id="ctx",
            task_id="TASK",
            candidate_tasks={f"candidate-{index}": "TASK" for index in range(5)},
            query_artifact_digest="d" * 64,
            membership_digest="e" * 64,
        )

    _write_raw_response(artifact := tmp_path / "production-response.json", binding)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["synthetic_fixture_only"] = False
    artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    non_fixture = RawDeltaTask5Adapter(
        SealedRawOperator(artifact, sha256_file(artifact)),
        method_id="RAW_ADAPTER_FIXTURE",
    )
    with pytest.raises(GateClosed, match="only an explicitly synthetic"):
        non_fixture.score(
            context_id="ctx",
            task_id="TASK",
            candidate_tasks={f"candidate-{index}": "TASK" for index in range(5)},
            query_artifact_digest="d" * 64,
            membership_digest="e" * 64,
        )


def test_production_raw_accepts_only_caller_digest_locked_external_response(
    tmp_path: Path,
):
    authority_digest = "a" * 64
    binding = _raw_request_binding()
    binding["method_id"] = RAW_PROJECT_METHOD_ID
    binding["operator_authority_sha256"] = authority_digest
    artifact = tmp_path / "external-response.json"
    _write_raw_response(
        artifact,
        binding,
        authority_digest=authority_digest,
    )
    adapter = RawDeltaTask5Adapter(
        SealedRawOperator(
            artifact,
            sha256_file(artifact),
            expected_authority_digest=authority_digest,
        ),
        method_id=RAW_PROJECT_METHOD_ID,
    )
    scores = adapter.score(
        context_id="ctx",
        task_id="TASK",
        candidate_tasks={f"candidate-{index}": "TASK" for index in range(5)},
        query_artifact_digest="d" * 64,
        membership_digest="e" * 64,
    )
    assert scores == {f"candidate-{index}": float(index) for index in range(5)}


@pytest.mark.parametrize(
    "tamper",
    [
        "response_bytes",
        "response_schema",
        "authority",
        "request_schema",
        "method_id",
        "task",
        "context",
        "candidate_ids",
        "query",
        "membership",
        "score_semantics",
        "missing_score_semantics",
        "fixture_flag",
        "extra_oracle",
    ],
)
def test_production_raw_external_response_tampering_fails_closed(
    tmp_path: Path,
    tamper: str,
):
    authority_digest = "a" * 64
    binding = _raw_request_binding()
    binding["method_id"] = RAW_PROJECT_METHOD_ID
    binding["operator_authority_sha256"] = authority_digest
    if tamper == "request_schema":
        binding["request_schema"] = "wrong-request-schema"
    elif tamper == "method_id":
        binding["method_id"] = RAW_FIXTURE_METHOD_ID
    elif tamper == "task":
        binding["task_id"] = "OTHER"
    elif tamper == "context":
        binding["context_id"] = "other-context"
    elif tamper == "candidate_ids":
        binding["candidate_ids"][0] = "intruder"
    elif tamper == "query":
        binding["query"]["artifact_sha256"] = "0" * 64
    elif tamper == "membership":
        binding["membership_digest"] = "0" * 64
    artifact = tmp_path / f"external-{tamper}.json"
    response_authority = "b" * 64 if tamper == "authority" else authority_digest
    _write_raw_response(
        artifact,
        binding,
        schema=("wrong-response-schema" if tamper == "response_schema" else RAW_RESPONSE_SCHEMA),
        authority_digest=response_authority,
    )
    if tamper == "score_semantics":
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["score_semantics"] = "lower_is_better_distance"
        artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    elif tamper == "missing_score_semantics":
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        del payload["score_semantics"]
        artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    elif tamper == "fixture_flag":
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["synthetic_fixture_only"] = 0
        artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    elif tamper == "extra_oracle":
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["oracle_label"] = "candidate-4"
        artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    expected_response_digest = sha256_file(artifact)
    operator = SealedRawOperator(
        artifact,
        expected_response_digest,
        expected_authority_digest=authority_digest,
    )
    if tamper == "response_bytes":
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["scores"]["candidate-0"] = 99.0
        artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    adapter = RawDeltaTask5Adapter(operator, method_id=RAW_PROJECT_METHOD_ID)
    with pytest.raises(
        GateClosed,
        match=(
            "digest mismatch|binding mismatch|schema mismatch|semantics mismatch|"
            "response fields differ|fixture flag"
        ),
    ):
        adapter.score(
            context_id="ctx",
            task_id="TASK",
            candidate_tasks={f"candidate-{index}": "TASK" for index in range(5)},
            query_artifact_digest="d" * 64,
            membership_digest="e" * 64,
        )


def _frozen_raw_execution_fixture(tmp_path: Path, monkeypatch):
    repo = tmp_path / "old-policy-repo"
    required_sources = (
        "server/repro_fpo_ppo_v04a/bpr_runner.py",
        "server/repro_fpo_ppo_v03/development_baseline_runner.py",
        "src/policy_learnware_v0/hashing.py",
        "src/policy_learnware_v0/rkme/reducer.py",
        "src/policy_learnware_v0/v03/canonicalization.py",
        "src/policy_learnware_v0/v04a/protocol.py",
    )
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    for index, relative in enumerate(required_sources):
        source = repo / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"FROZEN_SOURCE = {index}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "frozen Raw fixture"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    source_digests = {
        relative: sha256_file(repo / relative) for relative in required_sources
    }

    run_root = tmp_path / "raw-run"
    raw_view = run_root / "views" / "delta_action"
    source_root = raw_view / "source"
    source_root.mkdir(parents=True)
    canonicalizer = "1" * 64
    protocol = "2" * 64
    view_config = raw_view / "config.json"
    view_config.write_text(
        json.dumps(
            {
                "schema": "policy-learnware.v031-raw-transition-controls.v0",
                "view_id": "V_DELTA_ONLY",
                "canonicalizer_digest": canonicalizer,
                "protocol_id": protocol,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    run_config = run_root / "run_config.json"
    run_config.write_text(
        json.dumps({"schema": "frozen-run-config", "formal": True}, sort_keys=True),
        encoding="utf-8",
    )
    candidates = [f"candidate-{index}" for index in range(5)]
    source_rkme = {}
    for index, candidate in enumerate(candidates):
        source = source_root / f"{candidate}.npz"
        np.savez(source, frozen=np.asarray([index], dtype=np.int64))
        source_rkme[candidate] = sha256_file(source)

    raw_binding = {
        "config_sha256": sha256_file(view_config),
        "run_config_sha256": sha256_file(run_config),
        "canonicalizer_digest": canonicalizer,
        "protocol_id": protocol,
        "source_rkme_sha256": source_rkme,
    }
    asset_census = tmp_path / "asset-census.json"
    asset_census.write_text(
        json.dumps(
            {
                "schema": "policy-learnware.v04a-fixed-probe-run.v1",
                "stage": "asset-census",
                "status": "PASS",
                "raw_delta": raw_binding,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    raw_adapter = tmp_path / "raw-adapter.json"
    raw_adapter.write_text(
        json.dumps(
            {
                "schema": "policy-learnware.v04a-raw-delta-adapter.v1",
                "identity": "V031_SOURCE_ONLY_CANONICALIZER_REPLAY",
                "canonicalizer_digest": canonicalizer,
                "target_rows_read_during_fit": 0,
                "tasks": {"TASK": {"observation_dim": 2, "action_dim": 1}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    authority_value = {
        "schema": RAW_EXECUTION_AUTHORITY_SCHEMA,
        "method_id": RAW_PROJECT_METHOD_ID,
        "score_semantics": RAW_SCORE_SEMANTICS,
        "old_repo": {
            "commit": commit,
            "tree_digest": tree,
            "source_digests": source_digests,
        },
        "asset_census_sha256": sha256_file(asset_census),
        "raw_adapter_sha256": sha256_file(raw_adapter),
        "raw_view": {
            "view_id": "V_DELTA_ONLY",
            **raw_binding,
        },
    }
    authority = tmp_path / "raw-authority.json"
    authority.write_text(json.dumps(authority_value, sort_keys=True), encoding="utf-8")
    authority_digest = sha256_file(authority)

    membership_digest = "e" * 64
    query = tmp_path / "reward-free-query.npz"
    np.savez(
        query,
        observation=np.zeros((64, 2), dtype=np.float64),
        action=np.zeros((64, 1), dtype=np.float64),
        next_observation=np.ones((64, 2), dtype=np.float64),
        native_timestep=np.rint(np.linspace(0, 999, 64)).astype(np.int64),
        episode_offsets=np.asarray([0, 64], dtype=np.int64),
        membership_digest=np.asarray(membership_digest),
    )
    query_digest = sha256_file(query)
    request_adapter = RawDeltaTask5Adapter(
        SealedRawOperator(
            tmp_path / "not-yet-written-response.json",
            "f" * 64,
            expected_authority_digest=authority_digest,
        ),
        method_id=RAW_PROJECT_METHOD_ID,
    )
    request = request_adapter.request(
        context_id="ctx",
        task_id="TASK",
        candidate_tasks={candidate: "TASK" for candidate in candidates},
        query_artifact_digest=query_digest,
        membership_digest=membership_digest,
    )

    calls = {"verify": 0, "scores": 0}

    class FakeProbe:
        def __init__(self, **values):
            self.__dict__.update(values)

    class FakeRunner:
        RewardFreeProbe = FakeProbe

        @staticmethod
        def _verify_raw_binding(run, view, candidate_ids):
            calls["verify"] += 1
            assert run["raw_delta"] == raw_binding
            assert view == raw_view.resolve()
            assert candidate_ids == candidates

        @staticmethod
        def raw_delta_task5_scores(**values):
            calls["scores"] += 1
            assert values["raw_view_root"] == raw_view.resolve()
            assert values["candidate_ids"] == candidates
            assert values["probe"].probe_membership_digest == membership_digest
            assert not hasattr(values["probe"], "reward")
            return {candidate: -float(index) for index, candidate in enumerate(candidates)}

    monkeypatch.setattr(
        adapters_module,
        "_load_frozen_raw_runner",
        lambda verified_root, runner_path: (
            FakeRunner
            if verified_root == repo.resolve()
            and runner_path == (repo / required_sources[0]).resolve()
            else pytest.fail("unexpected frozen runner locator")
        ),
    )
    return {
        "authority_path": authority,
        "expected_authority_sha256": authority_digest,
        "repo_root": repo,
        "raw_view_root": raw_view,
        "asset_census_path": asset_census,
        "raw_adapter_path": raw_adapter,
        "query_path": query,
        "expected_query_sha256": query_digest,
        "request": request,
        "calls": calls,
        "authority_value": authority_value,
        "candidates": candidates,
        "membership_digest": membership_digest,
    }


def test_execute_frozen_raw_query_calls_verified_operator_and_returns_sealed_payload(
    tmp_path: Path, monkeypatch
):
    fixture = _frozen_raw_execution_fixture(tmp_path, monkeypatch)
    payload = execute_frozen_raw_query(**{key: value for key, value in fixture.items() if key not in {
        "calls", "authority_value", "candidates", "membership_digest"
    }})
    assert fixture["calls"] == {"verify": 1, "scores": 1}
    assert payload["schema"] == RAW_RESPONSE_SCHEMA
    assert payload["score_semantics"] == RAW_SCORE_SEMANTICS
    assert payload["synthetic_fixture_only"] is False
    assert payload["scores"] == {
        candidate: -float(index) for index, candidate in enumerate(fixture["candidates"])
    }
    response = tmp_path / "sealed-response.json"
    response.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    sealed = SealedRawOperator(
        response,
        sha256_file(response),
        expected_authority_digest=fixture["expected_authority_sha256"],
    )
    assert sealed.scores(fixture["request"]) == payload["scores"]


def test_frozen_raw_loader_rejects_foreign_cached_policy_dependency(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "authorized"
    runner = repo / "server" / "repro_fpo_ppo_v04a" / "bpr_runner.py"
    runner.parent.mkdir(parents=True)
    (repo / "src").mkdir()
    runner.write_text("VALUE = 1\n", encoding="utf-8")
    foreign = tmp_path / "foreign" / "policy_learnware_v0" / "cached.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("FOREIGN = True\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "policy_learnware_v0.foreign_cached",
        SimpleNamespace(__file__=str(foreign)),
    )

    with pytest.raises(GateClosed, match="came from another checkout"):
        adapters_module._load_frozen_raw_runner(repo.resolve(), runner.resolve())


@pytest.mark.parametrize("initial", [False, True])
def test_frozen_raw_loader_restores_bytecode_flag(
    tmp_path: Path, monkeypatch, initial: bool
) -> None:
    repo = tmp_path / "authorized"
    runner = repo / "server" / "repro_fpo_ppo_v04a" / "bpr_runner.py"
    runner.parent.mkdir(parents=True)
    (repo / "src").mkdir()
    runner.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "dont_write_bytecode", initial)

    loaded = adapters_module._load_frozen_raw_runner(
        repo.resolve(), runner.resolve()
    )

    assert loaded.VALUE == 1
    assert sys.dont_write_bytecode is initial


def test_execute_frozen_raw_query_rejects_reward_channel_before_operator(
    tmp_path: Path, monkeypatch
):
    fixture = _frozen_raw_execution_fixture(tmp_path, monkeypatch)
    query = fixture["query_path"]
    with np.load(query, allow_pickle=False) as clean:
        values = {key: np.asarray(clean[key]) for key in clean.files}
    np.savez(query, **values, reward=np.zeros(64))
    fixture["expected_query_sha256"] = sha256_file(query)
    fixture["request"]["query"]["artifact_sha256"] = fixture[
        "expected_query_sha256"
    ]
    with pytest.raises(GateClosed, match="reward-free query schema"):
        execute_frozen_raw_query(**{key: value for key, value in fixture.items() if key not in {
            "calls", "authority_value", "candidates", "membership_digest"
        }})
    assert fixture["calls"] == {"verify": 0, "scores": 0}


@pytest.mark.parametrize("tamper", ["identity", "operator_source", "rkme", "membership", "truthy"])
def test_execute_frozen_raw_query_tampering_fails_closed(
    tmp_path: Path, monkeypatch, tamper: str
):
    fixture = _frozen_raw_execution_fixture(tmp_path, monkeypatch)
    if tamper in {"identity", "truthy"}:
        authority = dict(fixture["authority_value"])
        if tamper == "identity":
            authority["method_id"] = RAW_FIXTURE_METHOD_ID
        else:
            authority["verified"] = True
        fixture["authority_path"].write_text(
            json.dumps(authority, sort_keys=True), encoding="utf-8"
        )
        fixture["expected_authority_sha256"] = sha256_file(fixture["authority_path"])
        fixture["request"]["operator_authority_sha256"] = fixture[
            "expected_authority_sha256"
        ]
    elif tamper == "operator_source":
        source = fixture["repo_root"] / "src/policy_learnware_v0/rkme/reducer.py"
        source.write_text("TAMPERED = True\n", encoding="utf-8")
    elif tamper == "rkme":
        source = (
            fixture["raw_view_root"]
            / "source"
            / f"{fixture['candidates'][0]}.npz"
        )
        source.write_bytes(source.read_bytes() + b"tamper")
    else:
        query = fixture["query_path"]
        with np.load(query, allow_pickle=False) as clean:
            values = {key: np.asarray(clean[key]) for key in clean.files}
        values["membership_digest"] = np.asarray("0" * 64)
        np.savez(query, **values)
        fixture["expected_query_sha256"] = sha256_file(query)
        fixture["request"]["query"]["artifact_sha256"] = fixture[
            "expected_query_sha256"
        ]
    with pytest.raises(GateClosed):
        execute_frozen_raw_query(**{key: value for key, value in fixture.items() if key not in {
            "calls", "authority_value", "candidates", "membership_digest"
        }})
    assert fixture["calls"] == {"verify": 0, "scores": 0}


def test_census_missing_dataset_is_stable_no_go(tmp_path: Path):
    report = census_real_assets(dataset_path=tmp_path / "absent.npz")
    assert report["status"] == "NO_GO"
    assert report["native_timestep_status"] == "NO_GO_MISSING_DATASET"
    assert report["exact_behavior_density"] is False
    assert report["oracle_per_step_rewards"] is False
    assert report["actor_authority_valid"] is False
    assert {
        "NO_GO_ASSET_ABI",
        "NO_GO_EXISTING_LOG_DENSITY",
        "NO_GO_ORACLE_DISCOUNTED_VALUE",
        "NO_GO_ACTOR_AUTHORITY",
    }.issubset(report["gates"])


def test_census_malformed_scalar_dataset_is_stable_no_go(tmp_path: Path):
    dataset = tmp_path / "scalar.npz"
    np.savez(
        dataset,
        observation=np.asarray(1.0),
        action=np.zeros((1, 1)),
        reward=np.zeros(1),
        next_observation=np.zeros((1, 1)),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
        dataset_cut=np.zeros(1, dtype=bool),
        episode_id=np.zeros(1, dtype=np.int64),
        episode_offsets=np.asarray([0, 1]),
    )
    report = census_real_assets(
        dataset_path=dataset,
        expected_dataset_digest=sha256_file(dataset),
    )
    assert report["status"] == "NO_GO"
    assert report["dataset_structure_status"] in {
        "NO_GO_INVALID_STRUCTURE",
        "NO_GO_UNREADABLE_DATASET",
    }
    assert "NO_GO_ASSET_ABI" in report["gates"]


def test_census_accepts_valid_native_subsample_but_not_truthy_capability_claims(
    tmp_path: Path,
):
    dataset = tmp_path / "native-subsample.npz"
    native = np.asarray([2, 5, 1, 7], dtype=np.int64)
    np.savez(
        dataset,
        observation=np.zeros((4, 2)),
        action=np.zeros((4, 1)),
        reward=np.zeros(4),
        next_observation=np.zeros((4, 2)),
        terminated=np.zeros(4, dtype=bool),
        truncated=np.zeros(4, dtype=bool),
        dataset_cut=np.zeros(4, dtype=bool),
        episode_id=np.asarray([0, 0, 1, 1]),
        episode_offsets=np.asarray([0, 2, 4]),
        native_timestep=native,
        timestep_provenance=np.asarray("native_indices"),
    )
    density = tmp_path / "density.json"
    density.write_text(
        json.dumps(
            {
                "exact_arbitrary_action_log_prob": True,
                "distribution": "claimed-gaussian",
                "action_transform": "claimed-identity",
            }
        ),
        encoding="utf-8",
    )
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"per_step_rewards": [[1.0, 2.0]]}), encoding="utf-8")
    authority = tmp_path / "actor.json"
    _authority(semantics="deterministic").to_json(authority)

    report = census_real_assets(
        dataset_path=dataset,
        oracle_path=oracle,
        density_manifest_path=density,
        actor_authority_path=authority,
        horizon=10,
        expected_dataset_digest=sha256_file(dataset),
        expected_oracle_digest=sha256_file(oracle),
        expected_density_manifest_digest=sha256_file(density),
        expected_actor_authority_digest=sha256_file(authority),
    )
    assert report["native_timestep_status"] == "EXPLICIT_NATIVE_INDICES"
    assert report["dataset_authority_status"] == "DIGEST_VERIFIED"
    assert report["exact_behavior_density_declared"] is True
    assert report["exact_behavior_density"] is False
    assert report["oracle_per_step_rewards_declared"] is True
    assert report["oracle_per_step_rewards"] is False
    assert report["actor_authority_declared_valid"] is True
    assert report["actor_authority_valid"] is False
    assert report["status"] == "NO_GO"


def test_census_accepts_complete_early_terminated_native_episodes(tmp_path: Path):
    dataset = tmp_path / "early-terminated.npz"
    np.savez(
        dataset,
        observation=np.zeros((5, 2)),
        action=np.zeros((5, 1)),
        reward=np.zeros(5),
        next_observation=np.zeros((5, 2)),
        terminated=np.asarray([False, True, False, False, True]),
        truncated=np.zeros(5, dtype=bool),
        dataset_cut=np.zeros(5, dtype=bool),
        episode_id=np.asarray([0, 0, 1, 1, 1]),
        episode_offsets=np.asarray([0, 2, 5]),
        native_timestep=np.asarray([0, 1, 0, 1, 2]),
        timestep_provenance=np.asarray("episode_offsets"),
    )
    report = census_real_assets(
        dataset_path=dataset,
        horizon=5,
        expected_dataset_digest=sha256_file(dataset),
    )
    assert report["native_timestep_status"] == "EXPLICIT_COMPLETE_EPISODES"
    assert report["dataset_structure_status"] == "PASS"
    assert report["dataset_authority_status"] == "DIGEST_VERIFIED"
    assert "NO_GO_ASSET_ABI" not in report["gates"]
    # Production still fails closed on the three unavailable capabilities.
    assert report["status"] == "NO_GO"


def test_census_derives_native_time_for_complete_early_terminated_episodes(
    tmp_path: Path,
):
    dataset = tmp_path / "early-derived.npz"
    np.savez(
        dataset,
        observation=np.zeros((5, 2)),
        action=np.zeros((5, 1)),
        reward=np.zeros(5),
        next_observation=np.zeros((5, 2)),
        terminated=np.asarray([False, True, False, False, True]),
        truncated=np.zeros(5, dtype=bool),
        dataset_cut=np.zeros(5, dtype=bool),
        episode_id=np.asarray([0, 0, 1, 1, 1]),
        episode_offsets=np.asarray([0, 2, 5]),
    )
    report = census_real_assets(
        dataset_path=dataset,
        horizon=5,
        expected_dataset_digest=sha256_file(dataset),
    )
    assert (
        report["native_timestep_status"]
        == "DERIVABLE_FROM_COMPLETE_EPISODE_OFFSETS"
    )
    assert report["dataset_structure_status"] == "PASS"
    assert "NO_GO_ASSET_ABI" not in report["gates"]


def test_census_accepts_short_dataset_cut_as_explicit_membership_boundary(
    tmp_path: Path,
):
    dataset = tmp_path / "short-cut.npz"
    np.savez(
        dataset,
        observation=np.zeros((3, 2)),
        action=np.zeros((3, 1)),
        reward=np.zeros(3),
        next_observation=np.zeros((3, 2)),
        terminated=np.zeros(3, dtype=bool),
        truncated=np.asarray([False, False, True]),
        dataset_cut=np.asarray([False, False, True]),
        episode_id=np.zeros(3, dtype=np.int64),
        episode_offsets=np.asarray([0, 3]),
        native_timestep=np.arange(3),
        timestep_provenance=np.asarray("episode_offsets"),
        truncation_reason=np.asarray(["none", "none", "dataset_cut"]),
    )
    report = census_real_assets(
        dataset_path=dataset,
        horizon=5,
        expected_dataset_digest=sha256_file(dataset),
    )
    assert report["native_timestep_status"] == "EXPLICIT_COMPLETE_EPISODES"
    assert report["dataset_structure_status"] == "PASS"
    assert "NO_GO_ASSET_ABI" not in report["gates"]


def test_census_never_upgrades_sample_ordinals_to_native_time(tmp_path: Path):
    rows = 64
    dataset = tmp_path / "sample-ordinals.npz"
    np.savez(
        dataset,
        observation=np.zeros((rows, 2)),
        action=np.zeros((rows, 1)),
        reward=np.zeros(rows),
        next_observation=np.zeros((rows, 2)),
        terminated=np.zeros(rows, dtype=bool),
        truncated=np.concatenate([np.zeros(rows - 1, dtype=bool), [True]]),
        dataset_cut=np.concatenate([np.zeros(rows - 1, dtype=bool), [True]]),
        episode_id=np.zeros(rows, dtype=np.int64),
        episode_offsets=np.asarray([0, rows]),
        native_timestep=np.arange(rows),
        timestep_provenance=np.asarray("sample_ordinal"),
        truncation_reason=np.asarray([*["none"] * (rows - 1), "dataset_cut"]),
    )
    report = census_real_assets(
        dataset_path=dataset,
        horizon=1000,
        expected_dataset_digest=sha256_file(dataset),
    )
    assert report["native_timestep_status"] == "INVALID_TIMESTEP_PROVENANCE"
    assert "NO_GO_ASSET_ABI" in report["gates"]


@pytest.mark.parametrize("corruption", ["nan", "row_count"])
def test_digest_verified_dataset_still_fails_invalid_structure(
    tmp_path: Path,
    corruption: str,
):
    dataset = tmp_path / f"invalid-{corruption}.npz"
    observation = np.zeros((3, 2))
    next_observation = np.zeros((3, 2))
    if corruption == "nan":
        observation[1, 0] = np.nan
    else:
        next_observation = next_observation[:2]
    np.savez(
        dataset,
        observation=observation,
        action=np.zeros((3, 1)),
        reward=np.zeros(3),
        next_observation=next_observation,
        terminated=np.zeros(3, dtype=bool),
        truncated=np.zeros(3, dtype=bool),
        dataset_cut=np.zeros(3, dtype=bool),
        episode_id=np.zeros(3, dtype=np.int64),
        episode_offsets=np.asarray([0, 3]),
        native_timestep=np.arange(3),
    )
    report = census_real_assets(
        dataset_path=dataset,
        horizon=3,
        expected_dataset_digest=sha256_file(dataset),
    )
    assert report["dataset_authority_status"] == "DIGEST_VERIFIED"
    assert report["native_timestep_status"] == "EXPLICIT_FULL_EPISODES"
    assert report["dataset_structure_status"] == "NO_GO_INVALID_STRUCTURE"
    assert "NO_GO_ASSET_ABI" in report["gates"]
    assert report["status"] == "NO_GO"


def test_census_strict_contract_rejects_terminal_before_group_end(tmp_path: Path):
    dataset = tmp_path / "terminal-mid-group.npz"
    np.savez(
        dataset,
        observation=np.zeros((3, 2)),
        action=np.zeros((3, 1)),
        reward=np.zeros(3),
        next_observation=np.zeros((3, 2)),
        terminated=np.asarray([True, False, False]),
        truncated=np.zeros(3, dtype=bool),
        dataset_cut=np.zeros(3, dtype=bool),
        episode_id=np.zeros(3, dtype=np.int64),
        episode_offsets=np.asarray([0, 3]),
        native_timestep=np.arange(3),
    )
    report = census_real_assets(
        dataset_path=dataset,
        horizon=3,
        expected_dataset_digest=sha256_file(dataset),
    )
    assert report["dataset_structure_status"] == "NO_GO_INVALID_DATA_CONTRACT"
    assert report["native_timestep_status"] == "INVALID_DATA_CONTRACT"
    assert "NO_GO_ASSET_ABI" in report["gates"]
