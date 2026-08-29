# Policy Learnware OPE v0.4b

This branch is the historical, pre-asset OPE companion baseline. It provides a
small NumPy implementation of five finite-horizon estimators, a reward-free Raw
fixture, ranking seals, metrics export, and a static production preflight. It
does **not** contain the later real-smoke bridge or any experiment assets.
The package version is `0.4.0b0`; its audited starting point is
`v04b@637e6650b5ae419919b9ea65137bdc896bbfd6be` (tree
`f9d4cc68794934ab1080b95af4cea776245199f7`).

## Status and scientific boundary

The synthetic command can finish as `TOY_MVP_PASS`; that status proves only
that the method-level fixture executed. The static `real-preflight` command is
expected to finish `NO_GO`. Neither an engineering test nor a toy result
overrides the final outcome of the v0.4b evolution line:

`FROZEN_SUPPORTING_STUDY_PRE_ORACLE_NO_GO`

The downstream supporting study stopped before oracle access for these
pre-registered reasons:

- Cheetah FQE: `NO_GO_FIT_DIVERGENCE`.
- Walker MB-FF: `NO_GO_MODEL_ROLLOUT_DIVERGENCE`.
- KMIFQE: `NO_GO_EXISTING_LOG_DENSITY_AND_TARGET_POLICY_SEMANTICS`.
- ETM: `NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT`.

`RAW_ADAPTER_FIXTURE` consumes fixture scores to test a reward-free,
digest-bound request/response path. Its score is not an OPE value, it is not a
production Raw-Delta/RKME operator, and it does not establish paper parity.
The fitted estimators are compact project adaptations/references, not official
paper reproductions. Method provenance is in [THIRD_PARTY.md](THIRD_PARTY.md).
The external final-study receipt is stored at the canonical logical path
`reports/ope/v04b-development/Policy_Learnware_v0.4b_OPE_Supporting_Study_Final.md`;
it is not a repository asset.

## External artifacts

No datasets, actors, oracle payloads, logs, wheels, or experiment runs are
tracked here. Active external paths resolve in this order:

1. an absolute CLI path;
2. `--artifacts-root` for a relative CLI path;
3. `RL_LEARNWARE_ARTIFACTS_ROOT`;
4. `<verified-source-checkout-parent>/artifacts` when running from this source
   checkout.

An installed/foreign layout without an explicit path or artifact root fails
closed. Historical receipts remain byte-identical; relocation is recorded
outside Git rather than by rewriting their embedded paths. Asset access is
logically no-write and digest-bound where supported; this branch does not claim
that source filesystem permissions are read-only. Artifact writes into any Git
worktree or bare repository are rejected.

Set the common environment variable to the sibling `artifacts/` directory. For
example, from this repository root:

```bash
export RL_LEARNWARE_ARTIFACTS_ROOT="$PWD/../artifacts"
policy-learnware-ope toy --output ope/toy/v04b-seed7 --seed 7
policy-learnware-ope real-preflight --output ope/preflight/v04b.json
policy-learnware-ope census --dataset ope/datasets/bank.npz \
  --output ope/census/v04b.json
```

An explicit root is equivalent for relative paths:

```bash
policy-learnware-ope toy --artifacts-root "$PWD/../artifacts" \
  --output ope/toy/v04b-seed7 --seed 7
```

## Install and verify

```bash
python -m pip install -e '.[test]'
pytest
```

`real-preflight` deliberately returns exit code 2 and records the missing
production authorities. This branch must not be used to claim a completed real
OPE experiment.
