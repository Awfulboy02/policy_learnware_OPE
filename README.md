# Policy Learnware OPE v0.41b

This branch is the compact source and method release for finite-horizon policy
selection. It provides executable NumPy implementations, ranking seals,
fail-closed production preflight, and synthetic reproduction tests. It does not
contain the later real-smoke runner or any tracked datasets, actor bundles,
oracles, run outputs, wheels, or release payloads.

The original source release is commit
`9be1d4c0b632ce2f54d0037036873cf6240da1e2`, tree
`45fa10cd91dbb25f86853440d512b4e100303d11`.

## Method scope

- `FH_FQE` is the finite-horizon ridge FQE reference.
- `FH_KMIFQE` is a B20 protocol adaptation with a nonlinear critic, local
  action-Hessian metric, bias/variance bandwidth update, replacement
  resampling, and logged-adjacent-action TD. It is not official-code or
  published-number parity.
- `ETM_MBOPE` is an ETM-RFF protocol adaptation with training-time Langevin
  negatives and an analytic gradient-penalty VJP. Its fixed RFF energy is not
  the official MLP and it is not paper parity.
- `DOPE_STYLE_MB_FF` and `AR_MBOPE` are project references/adaptations, not
  official DOPE or B06 reproductions.
- `RAW_ADAPTER_FIXTURE` is a sealed reward-free fixture ranking score. It is not
  a production Raw operator, and its score is not a `J_gamma` value; value
  MAE/RMSE are therefore not applicable.

Method IDs bind the executed gamma and horizon: the toy fixture emits
`..._G099_H5`, never `H1000`.

## External artifacts

No experiment payload is tracked in Git. Paths resolve in this order:

1. an absolute CLI path;
2. a relative path under explicit `--artifacts-root`;
3. a relative path under `RL_LEARNWARE_ARTIFACTS_ROOT`;
4. from a verified source checkout only, `<repository-parent>/artifacts`.

Relative paths may not escape the selected root. Installed or foreign layouts
fail closed for relative paths when neither an explicit root nor the environment
variable is supplied; configured roots must be absolute. Outputs inside any
detected Git worktree or bare repository are rejected. Canonical OPE assets live
under `artifacts/ope/`; the historical improvement report is external at the
canonical logical path
`reports/ope/v041b/Policy_Learnware_v0.41b_Improvement_Report.md`.

```bash
python -m pip install -e '.[test]'
export RL_LEARNWARE_ARTIFACTS_ROOT=/path/to/artifacts
policy-learnware-ope toy --output ope/toy/v041b-seed7 --seed 7
policy-learnware-ope real-preflight --output ope/preflight/v041b.json
policy-learnware-ope census \
  --dataset ope/datasets/example.npz \
  --output ope/census/example.json
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider
```

`TOY_MVP_PASS` and the original 94-test source-release audit mean that the
method-level fixture and engineering gates execute. They do not establish paper
parity or production validity.

## Scientific terminal state

The later v0.4b development line, not this branch, ran the two preregistered
real pre-oracle cells and froze the overall result as
`FROZEN_SUPPORTING_STUDY_PRE_ORACLE_NO_GO`:

- Cheetah FQE: `NO_GO_FIT_DIVERGENCE`;
- Walker MB-FF: `NO_GO_MODEL_ROLLOUT_DIVERGENCE`;
- KMIFQE: excluded as
  `NO_GO_EXISTING_LOG_DENSITY_AND_TARGET_POLICY_SEMANTICS` (missing exact
  existing-log density and incompatible stochastic-keyed target-policy
  semantics);
- ETM: excluded as `NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT`.

No oracle join was permitted. The engineering PASS and scientific NO-GO apply
to different scopes: source/wheel/toy PASS must never be read as overturning the
real-study terminal state.
