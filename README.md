# Policy Learnware OPE v0.4b development

This branch is the minimal companion implementation used for the completed
v0.4b supporting OPE study. Its scientific terminal state is
`FROZEN_SUPPORTING_STUDY_PRE_ORACLE_NO_GO`. Source, wheel, synthetic, and
fixture tests passing does not change that result.

The audited implementation tip before this maintenance pass is
`2743f2420840614e8ad4f07e69d8eaa744cdffc5` with tree
`c238a9feb69fde675f077fbf2d78b351e7f091bc`. It provides finite-horizon FQE,
the B20-inspired KMIFQE adaptation, compact model-based FF/AR and ETM
adaptations, digest-locked Raw execution, actor authority checks, ranking
seals, and an oracle-blind real-smoke runner. These are project adaptations,
not official-paper numerical reproductions.

## Frozen scientific result

The two preregistered development cells stopped before any oracle join:

- CheetahRun: FQE had genuine projected-Bellman fit divergence
  (`NO_GO_FIT_DIVERGENCE`). The four-candidate partial ranking is diagnostic
  only and is invalid for selection or metrics.
- WalkerWalk: MB-FF produced finite but physically divergent model rollouts
  (`NO_GO_MODEL_ROLLOUT_DIVERGENCE`). Its seal is retained only as failure
  evidence.
- KMIFQE was excluded as
  `NO_GO_EXISTING_LOG_DENSITY_AND_TARGET_POLICY_SEMANTICS`: the existing
  clipped-Gaussian log has no exact arbitrary-action density and the
  keyed-stochastic FPO target does not satisfy the original
  deterministic-target semantics.
- ETM was excluded by `NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT`.
- Raw emits a reward-free selection score (higher is better, negative MMD);
  that score is not a `J_gamma` value and must not be used for value-error
  metrics.

The real smoke remained oracle blind. Frozen receipts record
`oracle_accessed=false` and `environment_accessed=false`; no method subset
may be selected after the failures. The original plan called FPO deterministic,
but the frozen actors are per-step `STOCHASTIC_KEYED`; FQE and MB therefore
use candidate-independent common-random key panels, while KMIFQE fails closed.

The canonical final account is external at the logical path
`reports/ope/v04b-development/Policy_Learnware_v0.4b_OPE_Supporting_Study_Final.md`.
The v0.41b method/release account is external at the logical path
`reports/ope/v041b/Policy_Learnware_v0.41b_Improvement_Report.md`.

## External artifacts

No experiment dataset, actor bundle, run, oracle, wheel, or release receipt is
tracked here. The single asset root is selected in this order:

1. an absolute CLI/config path;
2. `--artifacts-root`;
3. `RL_LEARNWARE_ARTIFACTS_ROOT`;
4. for a verified source checkout only, `<repository-parent>/artifacts`.

Relative paths are confined below that root. Installed or foreign layouts
without an explicit root fail closed. The maintained OPE layout is:

```text
artifacts/
  ope/
    runs/v04b-real-smoke-2743f24-r0/
    releases/v04b-development-2743f24/
    releases/v041b-9be1d4c/
    receipts/v04b-development-2743f24/server-permission-freeze-v1/
  relocation_manifest.json
```

The real-run tree keeps all three immutable predecessor generations. Release
evidence keeps the complete nine-file inventory. Shared v03 datasets are
referenced through the v03 relocation manifest; oracle payloads are not copied
into OPE artifacts.

Historical absolute-path configs and receipts remain byte-identical and
continue to resolve as explicit paths. A relocated, relative config is a new
reconstructed invocation: it binds original asset digests and its new
implementation identity, while a separate external relocation manifest records
old-to-new paths and digests. A reconstructed checkout/runtime must report its
own verified commit/tree and dependencies; it must never claim to be the
original execution path or overwrite original provenance.

## Run and test

```bash
python -m pip install -e '.[test]'
export RL_LEARNWARE_ARTIFACTS_ROOT=/path/to/artifacts

policy-learnware-ope toy \
  --output ope/toy/seed-7 --seed 7
policy-learnware-ope real-preflight \
  --output ope/preflight/development.json
policy-learnware-ope real-smoke \
  --config ope/configs/reconstructed/<new-cell>.json \
  --expected-config-sha256 <sha256> \
  --output ope/runs/<new-no-clobber-run>

PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider
git diff --check
```

`toy` exercises five compact estimators plus the Raw fixture and may return
`TOY_MVP_PASS`. `real-preflight` is the retained legacy/static pre-asset
fail-closed check; it does not represent or replace the external final study
receipt. `real-smoke` is a no-clobber, resumable, pre-oracle bridge for Raw +
FQE + MB-FF only. Reusing it requires pinned census, membership, actor, bundle,
Raw authority, config, and implementation identities. Unknown programming
errors are not converted into scientific gate results.

The canonical frozen cell configs retain their original absolute-path
provenance and bytes. They are not rewritten after relocation. Any new
re-execution must use a new config under `ope/configs/reconstructed/`, a
separate sidecar classified as `RECONSTRUCTED_RELOCATION_CONFIG`, and a new
no-clobber output path. It cannot resume or claim the identity of the original
run. `real-preflight` checks only the static fail-closed status path;
`real-smoke` can exercise the pre-oracle bridge only when the separately pinned
runtime authorities and verified source checkouts are available. Neither path
accesses an oracle or changes the frozen scientific conclusion.

See `THIRD_PARTY.md` for upstream provenance and license boundaries.
