"""Digest-locked bridges to the frozen v03 repository and actor runtime.

This module deliberately contains no environment implementation, policy weights,
or Raw-RKME mathematics.  Production work is delegated to an authorized,
read-only process after its repository and source digests have been verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


class GateClosed(RuntimeError):
    """A scientific or provenance precondition failed closed."""

    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


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
            "repo_root": str(root),
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

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return sha256(payload).hexdigest()


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
        keys = None if action_keys is None else np.asarray(action_keys, dtype=np.uint64)
        if keys is None or keys.shape != (len(obs),):
            raise GateClosed(
                "NO_GO_TARGET_POLICY_SEMANTICS",
                "every actor query requires one explicit key per observation",
            )
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


class SubprocessActorProvider(InProcessActorProvider):
    """JSON-lines actor bridge; the service receives no env or oracle handle."""

    def __init__(
        self,
        authorities: Mapping[str, ActorAuthority],
        command: Sequence[str],
        *,
        cwd: str | Path,
        repo_authority: FrozenRepoAuthority,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.command = tuple(command)
        self.cwd = Path(cwd).resolve()
        self.repo_authority = repo_authority
        self.timeout_seconds = float(timeout_seconds)
        verified = self.repo_authority.verify()
        for authority in authorities.values():
            if authority.repo_commit != verified["commit"]:
                raise GateClosed("NO_GO_ASSET_ABI", "actor authority/repository commit mismatch")
            if authority.repo_tree_digest != verified["tree_digest"]:
                raise GateClosed("NO_GO_ASSET_ABI", "actor authority/repository tree mismatch")
        super().__init__(authorities, self._request)

    def _request(
        self,
        candidate_id: str,
        observations: np.ndarray,
        native_timestep: np.ndarray,
        keys: np.ndarray,
    ) -> np.ndarray:
        self.repo_authority.verify()
        authority = self.authorities[candidate_id]
        request = {
            "schema": authority.service_protocol,
            "candidate_id": candidate_id,
            "candidate_digest": authority.candidate_digest,
            "authority_sha256": authority.digest(),
            "observation_abi": authority.observation_abi,
            "action_abi": authority.action_abi,
            "observations": observations.tolist(),
            "native_timestep": [int(value) for value in native_timestep],
            "action_keys": [int(value) for value in keys],
        }
        completed = subprocess.run(
            list(self.command),
            cwd=self.cwd,
            input=json.dumps(request) + "\n",
            check=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        response = json.loads(completed.stdout)
        if response.get("candidate_id") != candidate_id:
            raise GateClosed("NO_GO_ASSET_ABI", "actor response candidate mismatch")
        if response.get("candidate_digest") != authority.candidate_digest:
            raise GateClosed("NO_GO_ASSET_ABI", "actor response digest mismatch")
        if response.get("authority_sha256") != authority.digest():
            raise GateClosed("NO_GO_ASSET_ABI", "actor response authority mismatch")
        if response.get("action_abi") != authority.action_abi:
            raise GateClosed("NO_GO_ASSET_ABI", "actor response action ABI mismatch")
        return np.asarray(response.get("actions"), dtype=float)

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


class SubprocessRawOperator:
    """Delegate Raw scoring to the frozen implementation without copying it."""

    def __init__(
        self,
        authority: FrozenRepoAuthority,
        command: Sequence[str],
        *,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.authority = authority
        self.command = tuple(command)
        self.timeout_seconds = float(timeout_seconds)

    def scores(self, request: Mapping[str, Any]) -> Mapping[str, float]:
        provenance = self.authority.verify()
        payload = dict(request)
        payload["authority"] = provenance
        completed = subprocess.run(
            list(self.command),
            cwd=self.authority.repo_root,
            input=json.dumps(payload) + "\n",
            check=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        response = json.loads(completed.stdout)
        return {str(key): float(value) for key, value in response["scores"].items()}


class SealedRawOperator:
    """Read an immutable Raw score artifact emitted by the frozen repository."""

    def __init__(
        self,
        artifact: str | Path,
        expected_digest: str,
        authority: FrozenRepoAuthority | None = None,
    ) -> None:
        self.artifact = Path(artifact)
        self.expected_digest = expected_digest
        self.authority = authority

    def scores(self, request: Mapping[str, Any]) -> Mapping[str, float]:
        if sha256_file(self.artifact) != self.expected_digest:
            raise GateClosed("NO_GO_ASSET_ABI", "sealed Raw artifact digest mismatch")
        payload = json.loads(self.artifact.read_text(encoding="utf-8"))
        if self.authority is None:
            if payload.get("synthetic_fixture_only") is not True:
                raise GateClosed(
                    "NO_GO_ASSET_ABI",
                    "production sealed Raw output requires frozen repository authority",
                )
        else:
            provenance = self.authority.verify()
            if payload.get("source_commit") != provenance["commit"]:
                raise GateClosed("NO_GO_ASSET_ABI", "sealed Raw source commit mismatch")
            if payload.get("source_tree_digest") != provenance["tree_digest"]:
                raise GateClosed("NO_GO_ASSET_ABI", "sealed Raw source tree mismatch")
        if payload.get("context_id") != request.get("context_id"):
            raise GateClosed("NO_GO_ASSET_ABI", "sealed Raw context mismatch")
        return {str(key): float(value) for key, value in payload["scores"].items()}


class RawDeltaTask5Adapter:
    """TASK_5 filter around an authorized Raw-Delta scoring delegate."""

    method_id = "RAW_DELTA_TASK5"

    def __init__(self, operator: RawOperator) -> None:
        self.operator = operator

    def score(
        self,
        *,
        context_id: str,
        task_id: str,
        candidate_tasks: Mapping[str, str],
        query_artifact: str,
        membership_digest: str,
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
        request = {
            "schema": "policy-learnware.raw-task5.v1",
            "method_id": self.method_id,
            "context_id": context_id,
            "task_id": task_id,
            "candidate_ids": eligible,
            "query_artifact": query_artifact,
            "membership_digest": membership_digest,
            "reward_visible": False,
            "candidate_actions_visible": False,
        }
        scores = dict(self.operator.scores(request))
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
) -> dict[str, Any]:
    """Perform a small read-only census and report gates instead of guessing."""

    dataset = Path(dataset_path)
    report: dict[str, Any] = {
        "dataset_path": str(dataset.resolve()),
        "horizon": int(horizon),
        "gates": [],
    }
    if not dataset.is_file():
        report["gates"].append("NO_GO_ASSET_ABI")
        report["dataset_error"] = "missing dataset"
        return report
    with np.load(dataset, allow_pickle=False) as data:
        keys = set(data.files)
        aliases = {
            "observation": {"observation", "observations"},
            "action": {"action", "actions"},
            "reward": {"reward", "rewards"},
            "next_observation": {"next_observation", "next_observations"},
            "terminated": {"terminated"},
            "truncated": {"truncated"},
            "episode_offsets": {"episode_offsets"},
        }
        missing = [name for name, options in aliases.items() if not (keys & options)]
        report["dataset_keys"] = sorted(keys)
        report["missing_fields"] = missing
        if missing:
            report["gates"].append("NO_GO_ASSET_ABI")
        offsets = np.asarray(data["episode_offsets"], dtype=int) if "episode_offsets" in data else None
        native = np.asarray(data["native_timestep"], dtype=int) if "native_timestep" in data else None
        provenance_value = data["timestep_provenance"] if "timestep_provenance" in data else None
        if provenance_value is not None:
            provenance_array = np.asarray(provenance_value)
            timestep_provenance = str(provenance_array.item()) if provenance_array.size == 1 else ""
        else:
            timestep_provenance = ""
        if offsets is None or offsets.ndim != 1 or len(offsets) < 2:
            native_status = "MISSING_EPISODE_OFFSETS"
        else:
            lengths = np.diff(offsets)
            full_episodes = bool(np.all(lengths == horizon))
            if native is not None:
                slices = [native[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]
                full_valid = len(native) == int(offsets[-1]) and full_episodes and all(
                    np.array_equal(values, np.arange(horizon)) for values in slices
                )
                sampled_valid = (
                    len(native) == int(offsets[-1])
                    and timestep_provenance == "native_indices"
                    and all(
                        len(values) > 1
                        and values[0] == 0
                        and values[-1] == horizon - 1
                        and np.all(np.diff(values) > 0)
                        for values in slices
                    )
                )
                if full_valid:
                    native_status = "EXPLICIT_FULL_EPISODES"
                elif sampled_valid:
                    native_status = "EXPLICIT_NATIVE_INDICES"
                else:
                    native_status = "INVALID_OR_COMPRESSED"
            elif full_episodes:
                native_status = "DERIVABLE_FROM_FULL_EPISODE_OFFSETS"
            else:
                native_status = "NO_GO_COMPRESSED_TIMESTEP"
        report["native_timestep_status"] = native_status
        if native_status not in {
            "EXPLICIT_FULL_EPISODES",
            "EXPLICIT_NATIVE_INDICES",
            "DERIVABLE_FROM_FULL_EPISODE_OFFSETS",
        }:
            report["gates"].append("NO_GO_ASSET_ABI")
    exact_density = False
    if density_manifest_path and Path(density_manifest_path).is_file():
        density = json.loads(Path(density_manifest_path).read_text(encoding="utf-8"))
        exact_density = bool(
            density.get("exact_arbitrary_action_log_prob")
            and density.get("distribution")
            and density.get("action_transform")
        )
    report["exact_behavior_density"] = exact_density
    if not exact_density:
        report["gates"].append("NO_GO_EXISTING_LOG_DENSITY")
    oracle_per_step = bool(oracle_path and _oracle_has_per_step_rewards(Path(oracle_path)))
    report["oracle_per_step_rewards"] = oracle_per_step
    if not oracle_per_step:
        report["gates"].append("NO_GO_ORACLE_DISCOUNTED_VALUE")
    actor_ok = False
    actor_semantics = None
    if actor_authority_path and Path(actor_authority_path).is_file():
        try:
            actor = ActorAuthority.from_json(actor_authority_path)
            actor_ok = True
            actor_semantics = actor.policy_semantics
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            report["actor_error"] = str(exc)
    report["actor_authority_valid"] = actor_ok
    report["actor_policy_semantics"] = actor_semantics
    if not actor_ok:
        report["gates"].append("NO_GO_ASSET_ABI")
    report["gates"] = sorted(set(report["gates"]))
    report["status"] = "PASS" if not report["gates"] else "FAIL_CLOSED"
    return report


__all__ = [
    "ActorAuthority",
    "ActorProvider",
    "BoundActorProvider",
    "FrozenRepoAuthority",
    "GateClosed",
    "InProcessActorProvider",
    "RawDeltaTask5Adapter",
    "SealedRawOperator",
    "SubprocessActorProvider",
    "SubprocessRawOperator",
    "census_real_assets",
    "sha256_file",
]
