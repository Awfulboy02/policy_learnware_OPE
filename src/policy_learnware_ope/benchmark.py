"""Oracle-blind ranking seals and post-seal OPE/selection metrics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
import csv
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


RANKING_SEAL_SCHEMA = "policy-learnware.ranking-seal.v2"
ORACLE_MANIFEST_SCHEMA = "policy-learnware.oracle-manifest.v1"


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _normalized_field_name(key: Any) -> str:
    return "".join(character for character in str(key).casefold() if character.isalnum())


def _assert_oracle_free(value: Any, *, path: str = "provenance") -> None:
    forbidden = (
        "oracle",
        "groundtruth",
        "truevalue",
        "truereturn",
        "truthvalue",
        "referencevalue",
        "referencereturn",
        "gtvalue",
        "gtreturn",
        "targetreturn",
        "episodereturn",
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_field_name(key)
            if any(token in normalized for token in forbidden):
                raise ValueError(f"oracle-bearing field cannot enter pre-oracle seal: {path}.{key}")
            _assert_oracle_free(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_oracle_free(item, path=f"{path}[{index}]")
    elif isinstance(value, (np.ndarray, np.generic)):
        _assert_oracle_free(_json_safe(value), path=path)
    elif isinstance(value, str):
        normalized = _normalized_field_name(value)
        if any(token in normalized for token in forbidden):
            raise ValueError(f"oracle-bearing value cannot enter pre-oracle seal: {path}")


def _assert_method_matches_value_convention(method_id: str, value_convention: str) -> None:
    method_match = re.search(r"_G([0-9]+)_H([1-9][0-9]*)(?:_|$)", method_id)
    if method_match is None:
        return
    convention_match = re.fullmatch(
        r"J_gamma=([0-9]+(?:\.[0-9]+)?)_H=([1-9][0-9]*)_raw",
        value_convention,
    )
    if convention_match is None:
        raise ValueError("finite-horizon method_id requires a parseable value convention")
    gamma = float(convention_match.group(1))
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("value convention gamma must lie in [0, 1]")
    if gamma == 1.0:
        gamma_token = "1000"
    else:
        text = np.format_float_positional(gamma, unique=True, trim="-")
        whole, _, fraction = text.partition(".")
        gamma_token = f"{whole}{fraction.ljust(2, '0')}"
    if (
        method_match.group(1) != gamma_token
        or method_match.group(2) != convention_match.group(2)
    ):
        raise ValueError("method_id gamma/horizon conflicts with value convention")


def _is_volatile_field(key: Any) -> bool:
    normalized = _normalized_field_name(key)
    explicit = any(
        token in normalized
        for token in (
            "runtime",
            "wallclock",
            "walltime",
            "elapsed",
            "duration",
            "latency",
            "fitseconds",
            "estimateseconds",
            "processtime",
            "timestamp",
            "startedat",
            "finishedat",
            "completedat",
        )
    )
    timing_unit = normalized.endswith(
        ("seconds", "milliseconds", "microseconds", "nanoseconds")
    )
    return explicit or timing_unit


def _without_volatile_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_volatile_fields(item)
            for key, item in value.items()
            if not _is_volatile_field(key)
        }
    if isinstance(value, (list, tuple)):
        return [_without_volatile_fields(item) for item in value]
    return _json_safe(value)


def _require_sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric and not boolean")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _normalized_candidate_ids(candidate_ids: Sequence[str]) -> list[str]:
    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError("candidate_ids must be a sequence of identifiers")
    normalized = sorted(str(candidate_id) for candidate_id in candidate_ids)
    if not normalized or any(not candidate_id for candidate_id in normalized):
        raise ValueError("candidate_ids must be non-empty identifiers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("candidate_ids must be unique after string normalization")
    return normalized


def candidate_set_digest(candidate_ids: Sequence[str]) -> str:
    """Return the canonical digest used to bind a candidate identity set."""

    normalized = _normalized_candidate_ids(candidate_ids)
    return _payload_digest(
        {
            "schema": "policy-learnware.candidate-set.v1",
            "candidate_ids": normalized,
        }
    )


def oracle_manifest_digest(oracle_manifest: Mapping[str, Any]) -> str:
    """Return the caller-held digest of a canonical oracle manifest."""

    if not isinstance(oracle_manifest, Mapping):
        raise TypeError("oracle_manifest must be a mapping")
    return _payload_digest(_json_safe(dict(oracle_manifest)))


@dataclass(frozen=True)
class RankingSeal:
    path: Path
    digest: str
    payload: Mapping[str, Any]


def seal_ranking(
    path: str | Path,
    *,
    method_id: str,
    context_id: str,
    score_kind: str,
    scores: Mapping[str, float | None],
    statuses: Mapping[str, str] | None = None,
    diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any],
    higher_is_better: bool = True,
    value_convention: str = "J_gamma=0.99_H=1000_raw",
) -> RankingSeal:
    """Write a canonical, oracle-blind ranking payload exactly once.

    The returned digest is the authority callers must retain separately.  It is
    deliberately not written into the seal file.
    """

    if score_kind not in {"value", "compatibility"}:
        raise ValueError("score_kind must be value or compatibility")
    if not method_id or not context_id or not value_convention or not scores:
        raise ValueError("method_id, context_id, value_convention, and scores are required")
    if not isinstance(higher_is_better, bool):
        raise TypeError("higher_is_better must be a boolean")
    _assert_method_matches_value_convention(str(method_id), str(value_convention))
    safe_provenance = _json_safe(dict(provenance))
    _assert_oracle_free(safe_provenance)
    raw_statuses = dict(statuses or {})
    raw_diagnostics = dict(diagnostics or {})
    scores_by_id = {str(candidate_id): score for candidate_id, score in scores.items()}
    statuses_by_id = {str(candidate_id): status for candidate_id, status in raw_statuses.items()}
    diagnostics_by_id = {
        str(candidate_id): candidate_diagnostics
        for candidate_id, candidate_diagnostics in raw_diagnostics.items()
    }
    if len(scores_by_id) != len(scores):
        raise ValueError("score candidate IDs collide after string normalization")
    if len(statuses_by_id) != len(raw_statuses) or len(diagnostics_by_id) != len(raw_diagnostics):
        raise ValueError("candidate IDs collide after string normalization")
    candidate_ids = _normalized_candidate_ids(list(scores_by_id))
    if not set(statuses_by_id).issubset(candidate_ids):
        raise ValueError("statuses contain candidates absent from scores")
    if not set(diagnostics_by_id).issubset(candidate_ids):
        raise ValueError("diagnostics contain candidates absent from scores")
    for candidate_id, candidate_diagnostics in diagnostics_by_id.items():
        if not isinstance(candidate_diagnostics, Mapping):
            raise TypeError(f"diagnostics for {candidate_id} must be a mapping")
        _assert_oracle_free(
            _json_safe(candidate_diagnostics),
            path=f"diagnostics[{candidate_id}]",
        )
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        raw_score = scores_by_id[candidate_id]
        score = (
            None
            if raw_score is None
            else _finite_number(raw_score, name=f"score for {candidate_id}")
        )
        status = statuses_by_id.get(candidate_id, "PASS" if score is not None else "INCOMPLETE")
        if score is None and status == "PASS":
            raise ValueError("PASS rows require a finite score")
        rows.append(
            {
                "candidate_id": candidate_id,
                "score": score,
                "status": str(status),
                "diagnostics": _without_volatile_fields(
                    _json_safe(diagnostics_by_id.get(candidate_id, {}))
                ),
            }
        )
    successful = [row for row in rows if row["score"] is not None and row["status"] == "PASS"]
    successful.sort(
        key=lambda row: (
            -row["score"] if higher_is_better else row["score"],
            row["candidate_id"],
        )
    )
    payload: dict[str, Any] = {
        "schema": RANKING_SEAL_SCHEMA,
        "method_id": str(method_id),
        "context_id": str(context_id),
        "score_kind": score_kind,
        "value_convention": str(value_convention),
        "candidate_set_digest": candidate_set_digest(candidate_ids),
        "higher_is_better": higher_is_better,
        "ranking": [row["candidate_id"] for row in successful],
        "selected_candidate_id": successful[0]["candidate_id"] if successful else None,
        "rows": rows,
        "provenance": _without_volatile_fields(safe_provenance),
    }
    digest = _payload_digest(payload)
    contents = _canonical_bytes(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as seal_file:
            seal_file.write(contents)
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file():
            raise FileExistsError(f"ranking seal path already exists and is not a regular file: {destination}")
        if destination.read_bytes() != contents:
            raise FileExistsError(f"ranking seal conflict; refusing to overwrite: {destination}")
    return RankingSeal(destination, digest, payload)


def _validate_ranking_payload(payload: Mapping[str, Any]) -> None:
    required_fields = {
        "schema",
        "method_id",
        "context_id",
        "score_kind",
        "value_convention",
        "candidate_set_digest",
        "higher_is_better",
        "ranking",
        "selected_candidate_id",
        "rows",
        "provenance",
    }
    if set(payload) != required_fields or payload.get("schema") != RANKING_SEAL_SCHEMA:
        raise ValueError("unsupported or malformed ranking seal schema")
    if not isinstance(payload.get("method_id"), str) or not payload["method_id"]:
        raise ValueError("ranking seal method_id is invalid")
    if not isinstance(payload.get("context_id"), str) or not payload["context_id"]:
        raise ValueError("ranking seal context_id is invalid")
    if payload.get("score_kind") not in {"value", "compatibility"}:
        raise ValueError("ranking seal score_kind is invalid")
    if not isinstance(payload.get("value_convention"), str) or not payload["value_convention"]:
        raise ValueError("ranking seal value_convention is invalid")
    _assert_method_matches_value_convention(payload["method_id"], payload["value_convention"])
    if not isinstance(payload.get("higher_is_better"), bool):
        raise ValueError("ranking seal higher_is_better is invalid")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ranking seal rows are invalid")
    candidate_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "candidate_id",
            "score",
            "status",
            "diagnostics",
        }:
            raise ValueError("ranking seal row is malformed")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("ranking seal candidate_id is invalid")
        candidate_ids.append(candidate_id)
        score = row.get("score")
        if score is not None and (
            not isinstance(score, (int, float)) or isinstance(score, bool) or not np.isfinite(score)
        ):
            raise ValueError(f"ranking seal score for {candidate_id} is invalid")
        if not isinstance(row.get("status"), str):
            raise ValueError(f"ranking seal status for {candidate_id} is invalid")
        if score is None and row["status"] == "PASS":
            raise ValueError("PASS rows require a finite score")
        if not isinstance(row.get("diagnostics"), Mapping):
            raise ValueError(f"ranking seal diagnostics for {candidate_id} are invalid")
        _assert_oracle_free(row["diagnostics"], path=f"diagnostics[{candidate_id}]")
        if _without_volatile_fields(row["diagnostics"]) != row["diagnostics"]:
            raise ValueError("ranking seal contains volatile diagnostics")
    if candidate_ids != _normalized_candidate_ids(candidate_ids):
        raise ValueError("ranking seal candidates are not unique and canonically ordered")
    if payload.get("candidate_set_digest") != candidate_set_digest(candidate_ids):
        raise ValueError("ranking seal candidate-set digest mismatch")
    successful = [row for row in rows if row["score"] is not None and row["status"] == "PASS"]
    successful.sort(
        key=lambda row: (
            -row["score"] if payload["higher_is_better"] else row["score"],
            row["candidate_id"],
        )
    )
    expected_ranking = [row["candidate_id"] for row in successful]
    if payload.get("ranking") != expected_ranking:
        raise ValueError("ranking seal ranking is inconsistent with its rows")
    expected_selected = expected_ranking[0] if expected_ranking else None
    if payload.get("selected_candidate_id") != expected_selected:
        raise ValueError("ranking seal selected candidate is inconsistent with its ranking")
    if not isinstance(payload.get("provenance"), Mapping):
        raise ValueError("ranking seal provenance is invalid")
    _assert_oracle_free(payload["provenance"])
    if _without_volatile_fields(payload["provenance"]) != payload["provenance"]:
        raise ValueError("ranking seal contains volatile provenance")


def load_ranking_seal(
    path: str | Path,
    *,
    expected_seal_digest: str,
) -> RankingSeal:
    """Load a seal only when it matches a digest held outside the seal file."""

    expected = _require_sha256(expected_seal_digest, name="expected_seal_digest")
    source = Path(path)
    contents = source.read_bytes()
    try:
        payload = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ranking seal is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("ranking seal must contain one payload object")
    digest = _payload_digest(payload)
    if not compare_digest(digest, expected):
        raise ValueError("ranking seal does not match caller-held expected digest")
    if contents != _canonical_bytes(payload):
        raise ValueError("ranking seal bytes are not canonical")
    _validate_ranking_payload(payload)
    return RankingSeal(source, digest, payload)


def _average_ranks(values: Sequence[float], *, descending: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(-array if descending else array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    position = 0
    while position < len(array):
        stop = position + 1
        while stop < len(array) and array[order[stop]] == array[order[position]]:
            stop += 1
        ranks[order[position:stop]] = (position + 1 + stop) / 2.0
        position = stop
    return ranks


def _rank_correlations(estimated: Sequence[float], oracle: Sequence[float]) -> tuple[float | None, float | None]:
    if len(estimated) < 2:
        return None, None
    est_ranks = _average_ranks(estimated, descending=True)
    oracle_ranks = _average_ranks(oracle, descending=True)
    if np.std(est_ranks) == 0 or np.std(oracle_ranks) == 0:
        spearman = None
    else:
        spearman = float(np.corrcoef(est_ranks, oracle_ranks)[0, 1])
    concordant = discordant = est_ties = oracle_ties = 0
    for left in range(len(estimated)):
        for right in range(left + 1, len(estimated)):
            est_delta = np.sign(estimated[left] - estimated[right])
            oracle_delta = np.sign(oracle[left] - oracle[right])
            if est_delta == 0 and oracle_delta == 0:
                continue
            if est_delta == 0:
                est_ties += 1
            elif oracle_delta == 0:
                oracle_ties += 1
            elif est_delta == oracle_delta:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + est_ties) * (concordant + discordant + oracle_ties)
    )
    kendall = None if denominator == 0 else float((concordant - discordant) / denominator)
    return spearman, kendall


def join_oracle_and_score(
    seal_path: str | Path,
    *,
    expected_seal_digest: str,
    oracle_manifest: Mapping[str, Any],
    expected_oracle_digest: str,
    winner_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Join an externally authenticated oracle manifest to an authenticated seal.

    ``oracle_manifest`` uses ``policy-learnware.oracle-manifest.v1`` and has
    five required fields: ``schema``, ``context_id``, ``candidate_values``,
    ``candidate_set_digest``, and ``value_convention``.  Optional normalized
    values must be supplied as the authenticated pair
    ``secondary_candidate_values`` and ``secondary_value_convention``.
    """

    seal = load_ranking_seal(seal_path, expected_seal_digest=expected_seal_digest)
    if not isinstance(oracle_manifest, Mapping):
        raise TypeError("oracle_manifest must be a mapping")
    manifest = _json_safe(dict(oracle_manifest))
    expected_oracle = _require_sha256(
        expected_oracle_digest,
        name="expected_oracle_digest",
    )
    actual_oracle = _payload_digest(manifest)
    if not compare_digest(actual_oracle, expected_oracle):
        raise ValueError("oracle manifest does not match caller-held expected digest")
    required_manifest_fields = {
        "schema",
        "context_id",
        "candidate_values",
        "candidate_set_digest",
        "value_convention",
    }
    optional_manifest_fields = {
        "secondary_candidate_values",
        "secondary_value_convention",
    }
    if not required_manifest_fields.issubset(manifest) or not set(manifest).issubset(
        required_manifest_fields | optional_manifest_fields
    ):
        raise ValueError("oracle manifest fields do not match its schema")
    if manifest.get("schema") != ORACLE_MANIFEST_SCHEMA:
        raise ValueError("unsupported oracle manifest schema")
    if manifest.get("context_id") != seal.payload["context_id"]:
        raise ValueError("oracle context differs from sealed context")
    if manifest.get("value_convention") != seal.payload["value_convention"]:
        raise ValueError("oracle value convention differs from sealed value convention")
    has_secondary_values = "secondary_candidate_values" in manifest
    has_secondary_convention = "secondary_value_convention" in manifest
    if has_secondary_values != has_secondary_convention:
        raise ValueError("secondary oracle values and convention must be supplied together")
    rows = list(seal.payload["rows"])
    candidate_ids = [row["candidate_id"] for row in rows]
    raw_oracle_values = manifest.get("candidate_values")
    if not isinstance(raw_oracle_values, Mapping):
        raise ValueError("oracle candidate_values must be a mapping")
    oracle_values = {str(key): value for key, value in raw_oracle_values.items()}
    if len(oracle_values) != len(raw_oracle_values):
        raise ValueError("oracle candidate IDs collide after string normalization")
    if set(candidate_ids) != set(oracle_values):
        raise ValueError("oracle candidate set differs from sealed candidate set")
    recomputed_candidate_digest = candidate_set_digest(list(oracle_values))
    if manifest.get("candidate_set_digest") != recomputed_candidate_digest:
        raise ValueError("oracle candidate-set digest does not match candidate_values")
    if not compare_digest(recomputed_candidate_digest, seal.payload["candidate_set_digest"]):
        raise ValueError("oracle candidate-set digest differs from sealed candidate set")
    oracle = {
        key: _finite_number(value, name=f"oracle value for {key}")
        for key, value in oracle_values.items()
    }
    tolerance = _finite_number(winner_tolerance, name="winner_tolerance")
    if tolerance < 0:
        raise ValueError("winner_tolerance must be finite and non-negative")
    best_value = max(oracle.values())
    winners = sorted(
        key for key, value in oracle.items() if best_value - value <= tolerance
    )
    selected = seal.payload["selected_candidate_id"]
    successful = [row for row in rows if row["status"] == "PASS" and row["score"] is not None]
    common_ids = [row["candidate_id"] for row in successful]
    estimated = [float(row["score"]) for row in successful]
    oracle_common = [oracle[candidate_id] for candidate_id in common_ids]
    oriented_estimated = (
        estimated if seal.payload["higher_is_better"] else [-value for value in estimated]
    )
    spearman, kendall = _rank_correlations(oriented_estimated, oracle_common)
    if seal.payload["score_kind"] == "value" and common_ids:
        errors = np.asarray(estimated) - np.asarray(oracle_common)
        value_mae: float | None = float(np.mean(np.abs(errors)))
        value_rmse: float | None = float(np.sqrt(np.mean(errors**2)))
        value_bias: float | None = float(np.mean(errors))
    else:
        value_mae = value_rmse = value_bias = None
    diagnostics = [row.get("diagnostics", {}) for row in rows]
    ess_values = [
        _finite_number(item["ess"], name="diagnostics.ess")
        for item in diagnostics
        if item.get("ess") is not None
    ]
    ess_fractions = [
        _finite_number(item["ess_fraction"], name="diagnostics.ess_fraction")
        for item in diagnostics
        if item.get("ess_fraction") is not None
    ]
    support_rows = [
        _finite_number(item["support_rows"], name="diagnostics.support_rows")
        for item in diagnostics
        if item.get("support_rows") is not None
    ]
    support_fields = {
        "ess",
        "ess_fraction",
        "support_rows",
        "mean_action_distance",
        "p95_action_distance",
        "max_action_distance",
        "rollout_action_z_mean",
        "rollout_action_z_max",
    }
    support_by_candidate = {
        row["candidate_id"]: {
            key: value
            for key, value in row.get("diagnostics", {}).items()
            if key in support_fields
        }
        for row in rows
    }
    selected_secondary = None
    secondary_value_convention = None
    if has_secondary_values:
        raw_secondary = manifest["secondary_candidate_values"]
        if not isinstance(raw_secondary, Mapping):
            raise ValueError("secondary_candidate_values must be a mapping")
        secondary_values = {
            str(key): _finite_number(value, name=f"secondary oracle value for {key}")
            for key, value in raw_secondary.items()
        }
        if len(secondary_values) != len(raw_secondary) or set(secondary_values) != set(candidate_ids):
            raise ValueError("secondary-value candidate set differs from seal")
        if not np.isfinite(list(secondary_values.values())).all():
            raise ValueError("secondary oracle values must be finite")
        secondary_value_convention = str(manifest["secondary_value_convention"])
        if not secondary_value_convention:
            raise ValueError("secondary_value_convention must be non-empty")
        if selected is not None:
            selected_secondary = secondary_values[selected]
    value_convention = seal.payload.get("value_convention")
    selected_primary = None if selected is None else oracle[selected]
    return {
        "schema": "policy-learnware.oracle-join.v2",
        "ranking_seal_sha256": seal.digest,
        "oracle_manifest_sha256": actual_oracle,
        "method_id": seal.payload["method_id"],
        "context_id": seal.payload["context_id"],
        "score_kind": seal.payload["score_kind"],
        "value_convention": value_convention,
        "selected_candidate_id": selected,
        "oracle_winner_ids": winners,
        "hit_at_1": None if selected is None else int(selected in winners),
        "regret_at_1": None if selected is None else float(best_value - oracle[selected]),
        "spearman": spearman,
        "kendall_tau_b": kendall,
        "value_mae": value_mae,
        "value_rmse": value_rmse,
        "value_bias": value_bias,
        "selected_primary_value": selected_primary,
        "secondary_value_convention": secondary_value_convention,
        "selected_secondary_value": selected_secondary,
        "selected_J_norm": (
            selected_secondary if secondary_value_convention == "J_norm" else None
        ),
        "ess_min": min(ess_values) if ess_values else None,
        "ess_mean": float(np.mean(ess_values)) if ess_values else None,
        "ess_fraction_min": min(ess_fractions) if ess_fractions else None,
        "support_rows_min": min(support_rows) if support_rows else None,
        "support_rows_mean": float(np.mean(support_rows)) if support_rows else None,
        "support_by_candidate": support_by_candidate,
        "runtime_seconds": None,
        "successful_candidates": len(successful),
        "candidate_count": len(rows),
        "status_coverage": float(len(successful) / len(rows)),
        "failure_coverage": float(1.0 - len(successful) / len(rows)),
        "failures": {
            row["candidate_id"]: row["status"] for row in rows if row["status"] != "PASS"
        },
        "provenance": seal.payload["provenance"],
    }


def export_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> dict[str, str]:
    """Write stable JSON and a compact flat summary CSV."""

    safe_records = [_json_safe(dict(record)) for record in records]
    json_payload = {"schema": "policy-learnware.metrics.v1", "records": safe_records}
    json_destination = Path(json_path)
    csv_destination = Path(csv_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_bytes(_canonical_bytes(json_payload))
    columns = [
        "method_id",
        "context_id",
        "score_kind",
        "value_convention",
        "value_mae",
        "value_rmse",
        "value_bias",
        "hit_at_1",
        "regret_at_1",
        "spearman",
        "kendall_tau_b",
        "ess_min",
        "ess_mean",
        "ess_fraction_min",
        "support_rows_min",
        "support_rows_mean",
        "runtime_seconds",
        "status_coverage",
        "failure_coverage",
        "ranking_seal_sha256",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for record in safe_records:
        writer.writerow({key: "" if record.get(key) is None else record.get(key) for key in columns})
    csv_destination.write_text(buffer.getvalue(), encoding="utf-8")
    return {
        "json": str(json_destination),
        "json_sha256": sha256(json_destination.read_bytes()).hexdigest(),
        "csv": str(csv_destination),
        "csv_sha256": sha256(csv_destination.read_bytes()).hexdigest(),
    }


__all__ = [
    "ORACLE_MANIFEST_SCHEMA",
    "RANKING_SEAL_SCHEMA",
    "RankingSeal",
    "candidate_set_digest",
    "export_metrics",
    "join_oracle_and_score",
    "load_ranking_seal",
    "oracle_manifest_digest",
    "seal_ranking",
]
