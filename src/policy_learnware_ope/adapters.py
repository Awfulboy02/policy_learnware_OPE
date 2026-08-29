"""Digest-locked bridges to the frozen v03 repository and actor runtime.

This module deliberately contains no environment implementation, policy weights,
or Raw-RKME mathematics. Only the sealed synthetic Raw adapter fixture is
executable in this release. Production Raw remains fail-closed until an
externally trusted export authority and reward-free membership verifier exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .core import DataValidationError, TransitionBatch, validate_action_keys


class GateClosed(RuntimeError):
    """A scientific or provenance precondition failed closed."""

    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


RAW_QUERY_SCHEMA = "policy-learnware.raw-query.reward-free.v1"
RAW_REQUEST_SCHEMA = "policy-learnware.raw-task5-request.v2"
RAW_RESPONSE_SCHEMA = "policy-learnware.raw-score-response.v2"
RAW_FIXTURE_METHOD_ID = "RAW_ADAPTER_FIXTURE"
RAW_PROJECT_METHOD_ID = "RAW_DELTA_TASK5_PROJECT_ADAPTER"
FROZEN_V03_COMMIT = "8b979f08c1d67e0eabfbda53b539ce67f21a6cfb"
FROZEN_V03_TREE = "7802977523fe2b0334b2f041a9fd8874e68f4aee"


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _finite_score_mapping(value: Any) -> dict[str, float]:
    """Normalize scores without allowing booleans or numeric strings."""

    if not isinstance(value, Mapping):
        raise GateClosed("NO_GO_ASSET_ABI", "Raw response scores must be an object")
    scores: dict[str, float] = {}
    for candidate_id, raw_score in value.items():
        if isinstance(raw_score, (bool, np.bool_)) or not isinstance(
            raw_score, (int, float, np.integer, np.floating)
        ):
            raise GateClosed(
                "NO_GO_ASSET_ABI",
                f"Raw score for {candidate_id!s} must be a JSON number",
            )
        score = float(raw_score)
        if not np.isfinite(score):
            raise GateClosed(
                "NO_GO_ASSET_ABI",
                f"Raw score for {candidate_id!s} is non-finite",
            )
        scores[str(candidate_id)] = score
    return scores


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenRepoAuthority:
    """Minimum authority binding for a read-only external implementation."""

    repo_root: Path
    commit: str
    source_digests: Mapping[str, str]
    tree_digest: str | None = None
    require_clean: bool = True

    def verify(self) -> dict[str, Any]:
        root = self.repo_root.resolve()
        if not (root / ".git").exists():
            raise GateClosed("NO_GO_ASSET_ABI", f"not a git repository: {root}")
        git_env = dict(os.environ)
        git_env["GIT_OPTIONAL_LOCKS"] = "0"
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
            env=git_env,
        ).stdout.strip()
        if actual != self.commit:
            raise GateClosed(
                "NO_GO_ASSET_ABI",
                f"repository commit drift: expected {self.commit}, got {actual}",
            )
        actual_tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
            env=git_env,
        ).stdout.strip()
        if self.tree_digest is not None and actual_tree != self.tree_digest:
            raise GateClosed(
                "NO_GO_ASSET_ABI",
                f"repository tree drift: expected {self.tree_digest}, got {actual_tree}",
            )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
            env=git_env,
        ).stdout
        if self.require_clean and status:
            raise GateClosed("NO_GO_ASSET_ABI", "authorized repository is not clean")
        checked: dict[str, str] = {}
        for relative, expected in sorted(self.source_digests.items()):
            source = (root / relative).resolve()
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise GateClosed("NO_GO_ASSET_ABI", f"source escapes repository: {relative}") from exc
            if not source.is_file():
                raise GateClosed("NO_GO_ASSET_ABI", f"missing authorized source: {relative}")
            actual_digest = sha256_file(source)
            if actual_digest != expected:
                raise GateClosed(
                    "NO_GO_ASSET_ABI",
                    f"source digest drift for {relative}: {actual_digest}",
                )
            checked[relative] = actual_digest
        return {
            "commit": actual,
            "tree_digest": actual_tree,
            "source_digests": checked,
        }

@dataclass(frozen=True)
class ActorAuthority:
    """Authority for one opaque candidate exposed by a read-only actor service."""

    candidate_id: str
    candidate_digest: str
    task_id: str
    observation_dim: int
    action_dim: int
    observation_abi: str
    action_abi: str
    policy_semantics: str
    normalizer_digest: str
    action_scaling_digest: str
    repo_commit: str
    repo_tree_digest: str
    upstream_runtime_commit: str
    source_digest: str
    dependency_lock_digest: str
    service_protocol: str = "policy-learnware.actor.v1"

    def __post_init__(self) -> None:
        if self.policy_semantics not in {"deterministic", "stochastic_keyed"}:
            raise ValueError("policy_semantics must be deterministic or stochastic_keyed")
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("actor dimensions must be positive")
        required = (
            self.candidate_id,
            self.candidate_digest,
            self.task_id,
            self.observation_abi,
            self.action_abi,
            self.normalizer_digest,
            self.action_scaling_digest,
            self.repo_commit,
            self.repo_tree_digest,
            self.upstream_runtime_commit,
            self.source_digest,
            self.dependency_lock_digest,
            self.service_protocol,
        )
        if any(not item for item in required):
            raise ValueError("actor authority fields must be non-empty")

    @classmethod
    def from_json(cls, path: str | Path) -> "ActorAuthority":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

class ActorProvider(Protocol):
    authorities: Mapping[str, ActorAuthority]

    def actions(
        self,
        candidate_id: str,
        observations: np.ndarray,
        *,
        native_timestep: np.ndarray,
        action_keys: np.ndarray | None,
        require_deterministic: bool = False,
    ) -> np.ndarray: ...


class InProcessActorProvider:
    """Key-explicit provider used by fixtures and trusted isolated adapters."""

    def __init__(
        self,
        authorities: Mapping[str, ActorAuthority],
        action_fn: Callable[[str, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self.authorities = dict(authorities)
        self._action_fn = action_fn

    def actions(
        self,
        candidate_id: str,
        observations: np.ndarray,
        *,
        native_timestep: np.ndarray,
        action_keys: np.ndarray | None,
        require_deterministic: bool = False,
    ) -> np.ndarray:
        if candidate_id not in self.authorities:
            raise GateClosed("NO_GO_ASSET_ABI", f"unknown candidate {candidate_id}")
        authority = self.authorities[candidate_id]
        obs = np.asarray(observations, dtype=float)
        if obs.ndim != 2 or obs.shape[1] != authority.observation_dim:
            raise GateClosed(
                "NO_GO_ASSET_ABI",
                f"{candidate_id} expects observation shape (*,{authority.observation_dim})",
            )
        times = np.asarray(native_timestep)
        if times.shape != (len(obs),) or times.dtype.kind not in "iu" or np.any(times < 0):
            raise GateClosed(
                "NO_GO_ASSET_ABI",
                "native_timestep must be one non-negative integer per row",
            )
        times = times.astype(np.int64)
        if action_keys is None:
            raise GateClosed(
                "NO_GO_TARGET_POLICY_SEMANTICS",
                "every actor query requires one explicit key per observation",
            )
        try:
            keys = validate_action_keys(action_keys, len(obs))
        except DataValidationError as exc:
            raise GateClosed(exc.status, exc.detail) from exc
        if authority.policy_semantics == "stochastic_keyed":
            if require_deterministic:
                raise GateClosed(
                    "NO_GO_TARGET_POLICY_SEMANTICS",
                    "baseline requires a deterministic evaluation policy",
                )
        actions = np.asarray(self._action_fn(candidate_id, obs, times, keys), dtype=float)
        if actions.shape != (len(obs), authority.action_dim) or not np.isfinite(actions).all():
            raise GateClosed("NO_GO_ASSET_ABI", "actor returned invalid action batch")
        return actions

    def bind(self, candidate_id: str) -> "BoundActorProvider":
        return BoundActorProvider(self, candidate_id)


class BoundActorProvider:
    """Expose one multi-candidate service through the estimator core protocol."""

    def __init__(self, provider: ActorProvider, candidate_id: str) -> None:
        if candidate_id not in provider.authorities:
            raise GateClosed("NO_GO_ASSET_ABI", f"unknown candidate {candidate_id}")
        self._provider = provider
        self._authority = provider.authorities[candidate_id]
        self.policy_id = candidate_id
        self.semantics = self._authority.policy_semantics

    def sample_actions(
        self,
        observations: np.ndarray,
        native_timestep: np.ndarray,
        *,
        keys: np.ndarray,
    ) -> np.ndarray:
        return self._provider.actions(
            self.policy_id,
            observations,
            native_timestep=native_timestep,
            action_keys=keys,
            require_deterministic=False,
        )


class RawOperator(Protocol):
    def scores(self, request: Mapping[str, Any]) -> Mapping[str, float]: ...


class SealedRawOperator:
    """Read an immutable synthetic Raw adapter response fixture."""

    def __init__(
        self,
        artifact: str | Path,
        expected_digest: str,
    ) -> None:
        self.artifact = Path(artifact)
        self.expected_digest = expected_digest

    def scores(self, request: Mapping[str, Any]) -> Mapping[str, float]:
        if sha256_file(self.artifact) != self.expected_digest:
            raise GateClosed("NO_GO_ASSET_ABI", "sealed Raw artifact digest mismatch")
        payload = json.loads(self.artifact.read_text(encoding="utf-8"))
        if payload.get("schema") != RAW_RESPONSE_SCHEMA:
            raise GateClosed("NO_GO_ASSET_ABI", "sealed Raw response schema mismatch")
        if payload.get("synthetic_fixture_only") is not True:
            raise GateClosed(
                "NO_GO_RAW_OPERATOR_AUTHORITY",
                "this release accepts only an explicitly synthetic Raw fixture response",
            )
        if request.get("method_id") != RAW_FIXTURE_METHOD_ID:
            raise GateClosed(
                "NO_GO_RAW_OPERATOR_AUTHORITY",
                "the sealed fixture may only use RAW_ADAPTER_FIXTURE",
            )
        binding = {
            "request_schema": request.get("schema"),
            "method_id": request.get("method_id"),
            "task_id": request.get("task_id"),
            "context_id": request.get("context_id"),
            "candidate_ids": request.get("candidate_ids"),
            "query": request.get("query"),
            "membership_digest": request.get("membership_digest"),
        }
        if payload.get("request_binding") != binding:
            raise GateClosed("NO_GO_ASSET_ABI", "sealed Raw request binding mismatch")
        if payload.get("request_sha256") != _canonical_digest(binding):
            raise GateClosed("NO_GO_ASSET_ABI", "sealed Raw request digest mismatch")
        scores = _finite_score_mapping(payload.get("scores"))
        if set(scores) != set(request.get("candidate_ids", [])):
            raise GateClosed("NO_GO_ASSET_ABI", "sealed Raw candidate membership mismatch")
        if not np.isfinite(list(scores.values())).all():
            raise GateClosed("NO_GO_ASSET_ABI", "sealed Raw response contains non-finite scores")
        return scores


class RawDeltaTask5Adapter:
    """Reward-free TASK_5 request binder around an authorized Raw delegate.

    This class contains no Raw/RKME scoring implementation.  A fixture must use
    ``RAW_ADAPTER_FIXTURE``; a production adapter should identify the concrete
    project adaptation and supply a digest-locked external operator.
    """

    def __init__(
        self,
        operator: RawOperator,
        *,
        method_id: str = RAW_PROJECT_METHOD_ID,
    ) -> None:
        self.operator = operator
        if not method_id:
            raise ValueError("method_id must be non-empty")
        self.method_id = str(method_id)
        if self.method_id not in {RAW_FIXTURE_METHOD_ID, RAW_PROJECT_METHOD_ID}:
            raise GateClosed(
                "NO_GO_RAW_OPERATOR_AUTHORITY",
                f"unsupported Raw method identity: {self.method_id}",
            )
        if self.method_id == RAW_PROJECT_METHOD_ID:
            raise GateClosed(
                "NO_GO_RAW_OPERATOR_AUTHORITY",
                "production Raw is deferred pending a trusted frozen-v03 export authority",
            )

    def score(
        self,
        *,
        context_id: str,
        task_id: str,
        candidate_tasks: Mapping[str, str],
        query_artifact_digest: str,
        membership_digest: str,
        query_schema: str = RAW_QUERY_SCHEMA,
    ) -> dict[str, float]:
        eligible = sorted(
            candidate_id
            for candidate_id, candidate_task in candidate_tasks.items()
            if candidate_task == task_id
        )
        if len(eligible) != 5:
            raise GateClosed(
                "NO_GO_ASSET_ABI",
                f"TASK_5 requires exactly five same-task candidates, got {len(eligible)}",
            )
        if query_schema != RAW_QUERY_SCHEMA:
            raise GateClosed("NO_GO_ASSET_ABI", "Raw query is not the reward-free query schema")
        if not _is_sha256(query_artifact_digest):
            raise GateClosed("NO_GO_ASSET_ABI", "Raw query artifact requires a SHA-256 digest")
        if not _is_sha256(membership_digest):
            raise GateClosed("NO_GO_ASSET_ABI", "Raw membership requires a SHA-256 digest")
        request = {
            "schema": RAW_REQUEST_SCHEMA,
            "method_id": self.method_id,
            "context_id": context_id,
            "task_id": task_id,
            "candidate_ids": eligible,
            "membership_digest": membership_digest,
            "query": {
                "schema": RAW_QUERY_SCHEMA,
                "artifact_sha256": query_artifact_digest,
                "fields": [
                    "observation",
                    "action",
                    "next_observation",
                    "native_timestep",
                    "episode_offsets",
                ],
                "forbidden_fields": ["reward", "oracle", "candidate_action"],
            },
        }
        scores = _finite_score_mapping(self.operator.scores(request))
        if set(scores) != set(eligible):
            raise GateClosed("NO_GO_ASSET_ABI", "Raw delegate returned a non-TASK_5 score set")
        if not np.isfinite(list(scores.values())).all():
            raise GateClosed("NO_GO_ASSET_ABI", "Raw delegate returned non-finite scores")
        return scores


def _oracle_has_per_step_rewards(path: Path) -> bool:
    if not path.exists():
        return False
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return any(
                key in data and np.asarray(data[key]).ndim >= 2
                for key in ("episode_rewards", "per_step_rewards", "rewards")
            )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("episode_rewards", "per_step_rewards", "rewards"):
        value = payload.get(key)
        if isinstance(value, list) and value and isinstance(value[0], list):
            return True
    return False


def census_real_assets(
    *,
    dataset_path: str | Path,
    oracle_path: str | Path | None = None,
    density_manifest_path: str | Path | None = None,
    actor_authority_path: str | Path | None = None,
    horizon: int = 1000,
    expected_dataset_digest: str | None = None,
    expected_oracle_digest: str | None = None,
    expected_density_manifest_digest: str | None = None,
    expected_actor_authority_digest: str | None = None,
) -> dict[str, Any]:
    """Perform a read-only capability census without trusting declarations.

    Paths are operational inputs, not reproduction identities, so the report
    records only content digests.  A digest can authenticate the inspected
    bytes; it cannot by itself prove a declared density, oracle convention, or
    actor implementation correct.  Those capabilities therefore remain
    ``DECLARED_ONLY`` until an executable authority checker is integrated.
    """

    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) <= 0:
        raise ValueError("horizon must be a positive integer")
    dataset = Path(dataset_path)
    report: dict[str, Any] = {
        "schema": "policy-learnware.real-asset-census.v2",
        "horizon": int(horizon),
        "gates": [],
        "dataset_present": dataset.is_file(),
        "dataset_sha256": sha256_file(dataset) if dataset.is_file() else None,
        "dataset_authority_status": "DECLARED_ONLY",
        "native_timestep_status": "NO_GO_MISSING_DATASET",
        "exact_behavior_density": False,
        "exact_behavior_density_status": "NO_GO_MISSING_DECLARATION",
        "oracle_per_step_rewards": False,
        "oracle_per_step_rewards_status": "NO_GO_MISSING_ORACLE",
        "actor_authority_valid": False,
        "actor_authority_status": "NO_GO_MISSING_AUTHORITY",
        "actor_policy_semantics": None,
    }
    if expected_dataset_digest is not None:
        if not _is_sha256(expected_dataset_digest):
            raise ValueError("expected_dataset_digest must be a lowercase SHA-256 digest")
        if report["dataset_sha256"] == expected_dataset_digest:
            report["dataset_authority_status"] = "DIGEST_VERIFIED"
        else:
            report["dataset_authority_status"] = "NO_GO_DIGEST_MISMATCH"
    if not dataset.is_file():
        report["gates"].append("NO_GO_ASSET_ABI")
        report["dataset_error"] = "missing dataset"
    else:
        try:
            with np.load(dataset, allow_pickle=False) as data:
                keys = set(data.files)
                aliases = {
                    "observation": {"observation", "observations"},
                    "action": {"action", "actions"},
                    "reward": {"reward", "rewards"},
                    "next_observation": {"next_observation", "next_observations"},
                    "terminated": {"terminated"},
                    "truncated": {"truncated"},
                    "dataset_cut": {"dataset_cut"},
                    "episode_id": {"episode_id"},
                    "episode_offsets": {"episode_offsets"},
                }
                missing = [name for name, options in aliases.items() if not (keys & options)]
                report["dataset_keys"] = sorted(keys)
                report["missing_fields"] = missing
                row_count: int | None = None
                structure_errors: list[str] = []
                observation = action = next_observation = np.empty((0, 1))
                reward = np.empty(0)
                terminated = truncated = dataset_cut = np.empty(0, dtype=bool)
                episode_id = np.empty(0, dtype=np.int64)
                if missing:
                    report["gates"].append("NO_GO_ASSET_ABI")
                    report["dataset_structure_status"] = "NO_GO_MISSING_FIELDS"
                else:
                    resolved = {
                        name: next(key for key in sorted(options) if key in keys)
                        for name, options in aliases.items()
                    }
                    observation = np.asarray(data[resolved["observation"]])
                    action = np.asarray(data[resolved["action"]])
                    reward = np.asarray(data[resolved["reward"]])
                    next_observation = np.asarray(data[resolved["next_observation"]])
                    terminated = np.asarray(data[resolved["terminated"]])
                    truncated = np.asarray(data[resolved["truncated"]])
                    dataset_cut = np.asarray(data[resolved["dataset_cut"]])
                    episode_id = np.asarray(data[resolved["episode_id"]])
                    if observation.ndim != 2 or observation.shape[1] == 0:
                        structure_errors.append("observation_shape")
                    if action.ndim != 2 or action.shape[1] == 0:
                        structure_errors.append("action_shape")
                    if next_observation.shape != observation.shape:
                        structure_errors.append("next_observation_shape")
                    if reward.ndim != 1:
                        structure_errors.append("reward_shape")
                    if terminated.ndim != 1 or truncated.ndim != 1:
                        structure_errors.append("termination_shape")
                    if reward.ndim == 1:
                        row_count = len(reward)
                        if any(
                            len(value) != row_count
                            for value in (
                                observation,
                                action,
                                next_observation,
                                terminated,
                                truncated,
                                dataset_cut,
                                episode_id,
                            )
                        ):
                            structure_errors.append("row_count")
                    numeric_values = (observation, action, reward, next_observation)
                    if any(
                        value.dtype.kind not in "biufc" or not np.isfinite(value).all()
                        for value in numeric_values
                    ):
                        structure_errors.append("non_finite_or_non_numeric")
                    for name, value in (
                        ("terminated", terminated),
                        ("truncated", truncated),
                        ("dataset_cut", dataset_cut),
                    ):
                        valid_boolean = value.dtype.kind == "b" or (
                            value.dtype.kind in "iu" and np.all((value == 0) | (value == 1))
                        )
                        if not valid_boolean:
                            structure_errors.append(f"{name}_not_boolean")
                    if episode_id.ndim != 1 or episode_id.dtype.kind not in "iu":
                        structure_errors.append("episode_id_not_integer_vector")
                    report["dataset_structure_errors"] = sorted(set(structure_errors))
                    report["dataset_structure_status"] = (
                        "PASS" if not structure_errors else "NO_GO_INVALID_STRUCTURE"
                    )
                    if structure_errors:
                        report["gates"].append("NO_GO_ASSET_ABI")
                offsets_raw = (
                    np.asarray(data["episode_offsets"])
                    if "episode_offsets" in data
                    else None
                )
                offsets = (
                    offsets_raw.astype(np.int64, copy=False)
                    if offsets_raw is not None and offsets_raw.dtype.kind in "iu"
                    else None
                )
                native = (
                    np.asarray(data["native_timestep"])
                    if "native_timestep" in data
                    else None
                )
                provenance_value = (
                    data["timestep_provenance"] if "timestep_provenance" in data else None
                )
                if provenance_value is not None:
                    provenance_array = np.asarray(provenance_value)
                    timestep_provenance = (
                        str(provenance_array.item()) if provenance_array.size == 1 else ""
                    )
                else:
                    timestep_provenance = ""
                provenance_allowed = timestep_provenance in {
                    "",
                    "episode_offsets",
                    "native_indices",
                }
                reasons = (
                    np.asarray(data["truncation_reason"])
                    if "truncation_reason" in data
                    else None
                )
                if not provenance_allowed:
                    native_status = "INVALID_TIMESTEP_PROVENANCE"
                elif (
                    offsets is None
                    or offsets.ndim != 1
                    or len(offsets) < 2
                    or offsets[0] != 0
                    or np.any(np.diff(offsets) <= 0)
                    or (row_count is not None and offsets[-1] != row_count)
                ):
                    native_status = "MISSING_OR_INVALID_EPISODE_OFFSETS"
                else:
                    lengths = np.diff(offsets)
                    full_episodes = bool(np.all(lengths == horizon))
                    if native is not None:
                        native_is_integer = native.ndim == 1 and native.dtype.kind in "iu"
                        slices = (
                            [native[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]
                            if native_is_integer and len(native) == int(offsets[-1])
                            else []
                        )
                        full_valid = bool(slices) and full_episodes and all(
                            np.array_equal(values, np.arange(horizon)) for values in slices
                        )
                        complete_episode_valid = (
                            bool(slices)
                            and timestep_provenance in {"", "episode_offsets"}
                            and len(terminated) == int(offsets[-1])
                            and len(truncated) == int(offsets[-1])
                            and all(
                                len(values) > 0
                                and len(values) <= horizon
                                and np.array_equal(values, np.arange(len(values)))
                                and (
                                    len(values) == horizon
                                    or bool(terminated[stop - 1])
                                    or (
                                        timestep_provenance == "episode_offsets"
                                        and bool(dataset_cut[stop - 1])
                                    )
                                    or (
                                        reasons is not None
                                        and reasons.ndim == 1
                                        and len(reasons) == int(offsets[-1])
                                        and str(reasons[stop - 1]) == "environment"
                                    )
                                )
                                for values, stop in zip(slices, offsets[1:])
                            )
                        )
                        # A legitimate native-index subsample need not contain
                        # either endpoint; it must only preserve physical index
                        # order and remain inside the declared horizon.
                        sampled_valid = (
                            bool(slices)
                            and timestep_provenance == "native_indices"
                            and all(
                                len(values) > 0
                                and np.all(values >= 0)
                                and np.all(values < horizon)
                                and np.all(np.diff(values) > 0)
                                for values in slices
                            )
                        )
                        if full_valid:
                            native_status = "EXPLICIT_FULL_EPISODES"
                        elif complete_episode_valid:
                            native_status = "EXPLICIT_COMPLETE_EPISODES"
                        elif sampled_valid:
                            native_status = "EXPLICIT_NATIVE_INDICES"
                        else:
                            native_status = "INVALID_OR_COMPRESSED"
                    elif timestep_provenance == "native_indices":
                        native_status = "NO_GO_MISSING_NATIVE_INDICES"
                    elif full_episodes:
                        native_status = "DERIVABLE_FROM_FULL_EPISODE_OFFSETS"
                    else:
                        complete_offsets = (
                            not missing
                            and len(terminated) == int(offsets[-1])
                            and all(
                                0 < int(length) <= horizon
                                and (
                                    int(length) == horizon
                                    or bool(terminated[stop - 1])
                                    or (
                                        timestep_provenance == "episode_offsets"
                                        and bool(dataset_cut[stop - 1])
                                    )
                                    or (
                                        reasons is not None
                                        and reasons.ndim == 1
                                        and len(reasons) == int(offsets[-1])
                                        and str(reasons[stop - 1]) == "environment"
                                    )
                                )
                                for length, stop in zip(lengths, offsets[1:])
                            )
                        )
                        native_status = (
                            "DERIVABLE_FROM_COMPLETE_EPISODE_OFFSETS"
                            if complete_offsets
                            else "NO_GO_COMPRESSED_TIMESTEP"
                        )
                if missing:
                    native_status = "NO_GO_MISSING_FIELDS"
                report["native_timestep_status"] = native_status
                if native_status not in {
                    "EXPLICIT_FULL_EPISODES",
                    "EXPLICIT_COMPLETE_EPISODES",
                    "EXPLICIT_NATIVE_INDICES",
                    "DERIVABLE_FROM_FULL_EPISODE_OFFSETS",
                    "DERIVABLE_FROM_COMPLETE_EPISODE_OFFSETS",
                }:
                    report["gates"].append("NO_GO_ASSET_ABI")
                if (
                    not missing
                    and not structure_errors
                    and offsets is not None
                    and native_status
                    in {
                        "EXPLICIT_FULL_EPISODES",
                        "EXPLICIT_COMPLETE_EPISODES",
                        "EXPLICIT_NATIVE_INDICES",
                        "DERIVABLE_FROM_FULL_EPISODE_OFFSETS",
                        "DERIVABLE_FROM_COMPLETE_EPISODE_OFFSETS",
                    }
                ):
                    try:
                        strict_native = native
                        strict_provenance = timestep_provenance
                        if strict_native is None:
                            strict_native = np.concatenate(
                                [np.arange(length, dtype=np.int64) for length in np.diff(offsets)]
                            )
                            strict_provenance = "episode_offsets"
                        elif native_status in {
                            "EXPLICIT_FULL_EPISODES",
                            "EXPLICIT_COMPLETE_EPISODES",
                        }:
                            strict_provenance = (
                                "native_indices"
                                if timestep_provenance == "native_indices"
                                else "episode_offsets"
                            )
                        elif native_status == "EXPLICIT_NATIVE_INDICES":
                            strict_provenance = "native_indices"
                        strict_batch = TransitionBatch(
                            observation=observation,
                            action=action,
                            reward=reward,
                            next_observation=next_observation,
                            terminated=terminated,
                            truncated=truncated,
                            dataset_cut=dataset_cut,
                            native_timestep=strict_native,
                            episode_id=episode_id,
                            episode_offsets=offsets,
                            timestep_provenance=strict_provenance,
                            truncation_reason=reasons,
                            source_digest=report["dataset_sha256"],
                        )
                        strict_batch.bootstrap_mask(int(horizon))
                    except (DataValidationError, ValueError) as exc:
                        report["dataset_structure_status"] = "NO_GO_INVALID_DATA_CONTRACT"
                        report["dataset_structure_errors"] = sorted(
                            set([*structure_errors, type(exc).__name__])
                        )
                        report["native_timestep_status"] = "INVALID_DATA_CONTRACT"
                        report["gates"].append("NO_GO_ASSET_ABI")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            report["dataset_error"] = f"unreadable dataset: {type(exc).__name__}"
            report.setdefault("dataset_structure_status", "NO_GO_UNREADABLE_DATASET")
            report["native_timestep_status"] = "NO_GO_UNREADABLE_DATASET"
            report["gates"].append("NO_GO_ASSET_ABI")

    if report["dataset_authority_status"] != "DIGEST_VERIFIED":
        report["gates"].append("NO_GO_ASSET_ABI")

    if density_manifest_path and Path(density_manifest_path).is_file():
        density_path = Path(density_manifest_path)
        report["density_manifest_sha256"] = sha256_file(density_path)
        try:
            density = json.loads(density_path.read_text(encoding="utf-8"))
            if not isinstance(density, Mapping):
                raise TypeError("density manifest must be an object")
            declaration = bool(
                density.get("exact_arbitrary_action_log_prob") is True
                and isinstance(density.get("distribution"), str)
                and density.get("distribution")
                and isinstance(density.get("action_transform"), str)
                and density.get("action_transform")
            )
            report["exact_behavior_density_declared"] = declaration
            digest_bound = (
                expected_density_manifest_digest is not None
                and _is_sha256(expected_density_manifest_digest)
                and report["density_manifest_sha256"] == expected_density_manifest_digest
            )
            report["exact_behavior_density_status"] = (
                "DECLARED_ONLY_DIGEST_BOUND" if declaration and digest_bound else "DECLARED_ONLY"
            )
        except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
            report["density_error"] = type(exc).__name__
            report["exact_behavior_density_status"] = "NO_GO_INVALID_DECLARATION"
    report["gates"].append("NO_GO_EXISTING_LOG_DENSITY")

    if oracle_path and Path(oracle_path).is_file():
        oracle = Path(oracle_path)
        report["oracle_sha256"] = sha256_file(oracle)
        try:
            declared_per_step = _oracle_has_per_step_rewards(oracle)
            report["oracle_per_step_rewards_declared"] = declared_per_step
            digest_bound = (
                expected_oracle_digest is not None
                and _is_sha256(expected_oracle_digest)
                and report["oracle_sha256"] == expected_oracle_digest
            )
            report["oracle_per_step_rewards_status"] = (
                "DECLARED_ONLY_DIGEST_BOUND"
                if declared_per_step and digest_bound
                else "DECLARED_ONLY"
            )
        except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
            report["oracle_error"] = type(exc).__name__
            report["oracle_per_step_rewards_status"] = "NO_GO_INVALID_ORACLE"
    report["gates"].append("NO_GO_ORACLE_DISCOUNTED_VALUE")

    if actor_authority_path and Path(actor_authority_path).is_file():
        actor_path = Path(actor_authority_path)
        report["actor_authority_sha256"] = sha256_file(actor_path)
        try:
            actor = ActorAuthority.from_json(actor_authority_path)
            report["actor_authority_declared_valid"] = True
            report["actor_policy_semantics"] = actor.policy_semantics
            digest_bound = (
                expected_actor_authority_digest is not None
                and _is_sha256(expected_actor_authority_digest)
                and report["actor_authority_sha256"] == expected_actor_authority_digest
            )
            report["actor_authority_status"] = (
                "DECLARED_ONLY_DIGEST_BOUND" if digest_bound else "DECLARED_ONLY"
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            report["actor_error"] = str(exc)
            report["actor_authority_status"] = "NO_GO_INVALID_DECLARATION"
    # Parsing a declaration is not an executable actor/repository authority
    # check, hence actor_authority_valid intentionally remains false.
    report["gates"].append("NO_GO_ACTOR_AUTHORITY")
    report["gates"] = sorted(set(report["gates"]))
    report["status"] = "NO_GO" if report["gates"] else "PASS"
    return report


__all__ = [
    "ActorAuthority",
    "ActorProvider",
    "BoundActorProvider",
    "FROZEN_V03_COMMIT",
    "FROZEN_V03_TREE",
    "FrozenRepoAuthority",
    "GateClosed",
    "InProcessActorProvider",
    "RAW_FIXTURE_METHOD_ID",
    "RAW_PROJECT_METHOD_ID",
    "RAW_QUERY_SCHEMA",
    "RAW_REQUEST_SCHEMA",
    "RAW_RESPONSE_SCHEMA",
    "RawDeltaTask5Adapter",
    "SealedRawOperator",
    "census_real_assets",
    "sha256_file",
]
