from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from policy_learnware_ope.benchmark import (
    ORACLE_MANIFEST_SCHEMA,
    RANKING_SEAL_SCHEMA,
    candidate_set_digest,
    join_oracle_and_score,
    load_ranking_seal,
    oracle_manifest_digest,
    seal_ranking,
)


VALUE_CONVENTION = "J_gamma=0.99_H=1000_raw"


def _canonical_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _seal(path: Path, **overrides):
    arguments = {
        "method_id": "FH_FQE_G099_H1000",
        "context_id": "context-7",
        "score_kind": "value",
        "scores": {"candidate-b": 1.0, "candidate-a": 2.5, "candidate-c": 0.0},
        "diagnostics": {"candidate-a": {"ess": 12.0}},
        "provenance": {"dataset_digest": "a" * 64, "seed": 17},
        "value_convention": VALUE_CONVENTION,
    }
    arguments.update(overrides)
    return seal_ranking(path, **arguments)


def _oracle_manifest(**overrides):
    values = {"candidate-a": 3.0, "candidate-b": 1.0, "candidate-c": 1.0}
    manifest = {
        "schema": ORACLE_MANIFEST_SCHEMA,
        "context_id": "context-7",
        "candidate_values": values,
        "candidate_set_digest": candidate_set_digest(list(values)),
        "value_convention": VALUE_CONVENTION,
    }
    manifest.update(overrides)
    return manifest


def _join(seal, manifest):
    return join_oracle_and_score(
        seal.path,
        expected_seal_digest=seal.digest,
        oracle_manifest=manifest,
        expected_oracle_digest=oracle_manifest_digest(manifest),
    )


def test_seal_is_payload_only_write_once_and_idempotent(tmp_path: Path):
    path = tmp_path / "ranking.json"
    first = _seal(path)
    original = path.read_bytes()

    document = json.loads(original)
    assert document["schema"] == RANKING_SEAL_SCHEMA
    assert "payload" not in document
    assert "sha256" not in document
    assert first.digest == sha256(original).hexdigest()

    second = _seal(path)
    assert second.digest == first.digest
    assert path.read_bytes() == original

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _seal(path, scores={"candidate-a": 99.0, "candidate-b": 1.0, "candidate-c": 0.0})
    assert path.read_bytes() == original


def test_runtime_and_wall_clock_fields_are_recursively_excluded_and_bytes_stable(
    tmp_path: Path,
):
    first = _seal(
        tmp_path / "first.json",
        diagnostics={
            "candidate-a": {
                "ess": 12.0,
                "runtime_seconds": 1.25,
                "fit_seconds": 3.0,
                "estimate_seconds": 4.0,
                "train_seconds": 8.0,
                "inference_milliseconds": 9.0,
                "nested": {
                    "wall-clock-ms": 44,
                    "wall_time_seconds": 45,
                    "solver": "fixed",
                },
            }
        },
        provenance={
            "dataset_digest": "a" * 64,
            "seed": 17,
            "timing": {"started_at": "now", "elapsed_seconds": 9.0},
        },
    )
    second = _seal(
        tmp_path / "second.json",
        diagnostics={
            "candidate-a": {
                "ess": 12.0,
                "runtime_seconds": 999.0,
                "fit_seconds": 30.0,
                "estimate_seconds": 40.0,
                "train_seconds": 80.0,
                "inference_milliseconds": 90.0,
                "nested": {
                    "wall-clock-ms": 0,
                    "wall_time_seconds": 450,
                    "solver": "fixed",
                },
            }
        },
        provenance={
            "dataset_digest": "a" * 64,
            "seed": 17,
            "timing": {"started_at": "later", "elapsed_seconds": 123.0},
        },
    )

    assert first.digest == second.digest
    assert first.path.read_bytes() == second.path.read_bytes()
    payload_text = first.path.read_text(encoding="utf-8").casefold()
    assert "runtime" not in payload_text
    assert "wall-clock" not in payload_text
    assert "wall_time" not in payload_text
    assert "fit_seconds" not in payload_text
    assert "estimate_seconds" not in payload_text
    assert "train_seconds" not in payload_text
    assert "inference_milliseconds" not in payload_text
    assert "elapsed" not in payload_text
    assert first.payload["rows"][0]["diagnostics"] == {"ess": 12.0, "nested": {"solver": "fixed"}}


@pytest.mark.parametrize(
    "container,leak",
    [
        ("provenance", {"nested": [{"ground-truth-return": 4.0}]}),
        ("provenance", {"nested": {"TRUE_VALUE": 4.0}}),
        ("provenance", {"nested": {"true_return": 4.0}}),
        ("provenance", {"nested": {"truth_value": 4.0}}),
        ("provenance", {"artifact": "/private/oracle/episode_returns.json"}),
        ("provenance", {"note": "selected from true return"}),
        ("diagnostics", {"candidate-a": {"deep": [{"oracleEstimate": 4.0}]}}),
        ("diagnostics", {"candidate-a": {"deep": {"episode_return": 4.0}}}),
    ],
)
def test_seal_recursively_rejects_oracle_class_fields(
    tmp_path: Path,
    container: str,
    leak: dict,
):
    with pytest.raises(ValueError, match="oracle-bearing"):
        _seal(tmp_path / f"{container}.json", **{container: leak})


@pytest.mark.parametrize("container", ["provenance", "diagnostics"])
def test_object_array_cannot_hide_oracle_fields(tmp_path: Path, container: str):
    hidden = np.asarray([{"oracle_value": 4.0}], dtype=object)
    leak = (
        {"hidden": hidden}
        if container == "provenance"
        else {"candidate-a": {"hidden": hidden}}
    )
    with pytest.raises(ValueError, match="oracle-bearing"):
        _seal(tmp_path / f"array-{container}.json", **{container: leak})


def test_method_identity_must_match_value_convention(tmp_path: Path):
    with pytest.raises(ValueError, match="conflicts with value convention"):
        _seal(
            tmp_path / "mislabeled.json",
            method_id="FH_FQE_G099_H1000",
            value_convention="J_gamma=0.99_H=5_raw",
        )


def test_non_numeric_scores_and_sort_direction_are_rejected(tmp_path: Path):
    with pytest.raises(TypeError, match="numeric and not boolean"):
        _seal(
            tmp_path / "bool-score.json",
            scores={"candidate-a": True, "candidate-b": 1.0},
        )
    with pytest.raises(TypeError, match="numeric and not boolean"):
        _seal(
            tmp_path / "string-score.json",
            scores={"candidate-a": "1.25", "candidate-b": 1.0},
        )
    with pytest.raises(TypeError, match="higher_is_better"):
        _seal(tmp_path / "bool-direction.json", higher_is_better="false")


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_values", True),
        ("candidate_values", "3.0"),
        ("secondary_candidate_values", "3.0"),
    ],
)
def test_oracle_join_rejects_non_numeric_values(
    tmp_path: Path,
    field: str,
    value: object,
):
    seal = _seal(tmp_path / f"oracle-{field}.json")
    manifest = _oracle_manifest()
    if field == "candidate_values":
        manifest[field]["candidate-a"] = value
    else:
        manifest[field] = dict(manifest["candidate_values"])
        manifest[field]["candidate-a"] = value
        manifest["secondary_value_convention"] = "J_norm"
    with pytest.raises(TypeError, match="numeric and not boolean"):
        _join(seal, manifest)


def test_oracle_join_rejects_boolean_winner_tolerance(tmp_path: Path):
    seal = _seal(tmp_path / "oracle-tolerance.json")
    manifest = _oracle_manifest()
    with pytest.raises(TypeError, match="winner_tolerance"):
        join_oracle_and_score(
            seal.path,
            expected_seal_digest=seal.digest,
            oracle_manifest=manifest,
            expected_oracle_digest=oracle_manifest_digest(manifest),
            winner_tolerance=True,
        )


def test_load_requires_external_digest_and_rejects_plain_tampering(tmp_path: Path):
    seal = _seal(tmp_path / "ranking.json")
    with pytest.raises(TypeError):
        load_ranking_seal(seal.path)  # type: ignore[call-arg]

    payload = json.loads(seal.path.read_bytes())
    payload["rows"][0]["score"] = 101.0
    seal.path.write_bytes(_canonical_bytes(payload))
    with pytest.raises(ValueError, match="caller-held expected digest"):
        load_ranking_seal(seal.path, expected_seal_digest=seal.digest)


def test_caller_digest_rejects_attacker_self_signed_replacement(tmp_path: Path):
    seal = _seal(tmp_path / "ranking.json")
    replacement = json.loads(seal.path.read_bytes())
    replacement["rows"][0]["score"] = 101.0
    replacement["sha256"] = sha256(_canonical_bytes(replacement)).hexdigest()
    seal.path.write_bytes(_canonical_bytes(replacement))

    with pytest.raises(ValueError, match="caller-held expected digest"):
        load_ranking_seal(seal.path, expected_seal_digest=seal.digest)


def test_authenticated_oracle_join_reports_metrics_without_sealed_runtime(tmp_path: Path):
    seal = _seal(
        tmp_path / "ranking.json",
        diagnostics={"candidate-a": {"ess": 12.0, "runtime_seconds": 3.0}},
    )
    manifest = _oracle_manifest(
        secondary_candidate_values={
            "candidate-a": 0.3,
            "candidate-b": 0.1,
            "candidate-c": 0.1,
        },
        secondary_value_convention="J_norm",
    )

    metrics = _join(seal, manifest)

    assert metrics["hit_at_1"] == 1
    assert metrics["value_mae"] == pytest.approx(0.5)
    assert metrics["ess_min"] == 12.0
    assert metrics["support_by_candidate"]["candidate-a"]["ess"] == 12.0
    assert metrics["status_coverage"] == 1.0
    assert metrics["failure_coverage"] == 0.0
    assert metrics["selected_J_norm"] == 0.3
    assert metrics["selected_secondary_value"] == 0.3
    assert metrics["secondary_value_convention"] == "J_norm"
    assert metrics["runtime_seconds"] is None
    assert metrics["ranking_seal_sha256"] == seal.digest
    assert metrics["oracle_manifest_sha256"] == oracle_manifest_digest(manifest)


def test_secondary_value_is_not_mislabeled_as_j_norm(tmp_path: Path):
    seal = _seal(tmp_path / "ranking.json")
    manifest = _oracle_manifest(
        secondary_candidate_values={
            "candidate-a": 99.0,
            "candidate-b": 10.0,
            "candidate-c": 1.0,
        },
        secondary_value_convention="authenticated_other_endpoint",
    )
    metrics = _join(seal, manifest)
    assert metrics["selected_secondary_value"] == 99.0
    assert metrics["secondary_value_convention"] == "authenticated_other_endpoint"
    assert metrics["selected_J_norm"] is None


@pytest.mark.parametrize(
    "override,error",
    [
        ({"schema": "policy-learnware.oracle-manifest.v0"}, "schema"),
        ({"context_id": "wrong-context"}, "context"),
        ({"value_convention": "J_gamma=1.0_H=1000_raw"}, "value convention"),
        ({"candidate_set_digest": "0" * 64}, "candidate-set digest"),
        (
            {
                "candidate_values": {
                    "candidate-a": 3.0,
                    "candidate-b": 1.0,
                    "intruder": 1.0,
                }
            },
            "candidate set",
        ),
    ],
)
def test_oracle_manifest_binding_rejects_wrong_authority(
    tmp_path: Path,
    override: dict,
    error: str,
):
    seal = _seal(tmp_path / "ranking.json")
    manifest = _oracle_manifest(**override)

    with pytest.raises(ValueError, match=error):
        _join(seal, manifest)


def test_join_rejects_wrong_external_seal_or_oracle_digest(tmp_path: Path):
    seal = _seal(tmp_path / "ranking.json")
    manifest = _oracle_manifest()

    with pytest.raises(ValueError, match="expected_seal_digest"):
        join_oracle_and_score(
            seal.path,
            expected_seal_digest="not-a-digest",
            oracle_manifest=manifest,
            expected_oracle_digest=oracle_manifest_digest(manifest),
        )
    with pytest.raises(ValueError, match="caller-held expected digest"):
        join_oracle_and_score(
            seal.path,
            expected_seal_digest=seal.digest,
            oracle_manifest=manifest,
            expected_oracle_digest="0" * 64,
        )


def test_oracle_manifest_cannot_be_mutated_after_its_digest_is_held(tmp_path: Path):
    seal = _seal(tmp_path / "ranking.json")
    manifest = _oracle_manifest()
    held_digest = oracle_manifest_digest(manifest)
    manifest["candidate_values"]["candidate-a"] = -100.0

    with pytest.raises(ValueError, match="caller-held expected digest"):
        join_oracle_and_score(
            seal.path,
            expected_seal_digest=seal.digest,
            oracle_manifest=manifest,
            expected_oracle_digest=held_digest,
        )
