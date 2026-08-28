# Policy Learnware OPE v0.41b

Small companion repository for method-level, finite-horizon policy-selection
experiments. The v03 repository, policy bundles, logs, manifests, and oracle
assets remain read-only external inputs.

The executable synthetic MVP contains five fitted estimators:

- `FH_FQE`: compact NumPy finite-horizon ridge FQE reference.
- `FH_KMIFQE`: B20 protocol adaptation with a candidate-specific nonlinear
  critic/target critic, local action-Hessian metrics, B20 bias/variance
  bandwidth updates, replacement importance resampling, and in-sample TD on
  logged adjacent actions.
- `ETM_MBOPE`: B22 protocol adaptation with model-generated training-time
  Langevin negatives and an analytic gradient-penalty VJP into the trainable
  RFF energy head.
- `DOPE_STYLE_MB_FF`: project-defined feed-forward model-based reference. DOPE
  is inspiration for the benchmark design, not the name of an upstream
  algorithm implemented here.
- `AR_MBOPE`: B06-inspired project adaptation/proxy, kept independent from the
  feed-forward model.

Method IDs are derived from the protocol actually executed. Thus the toy run
emits `..._G099_H5`; only a real `gamma=.99, H=1000` run may emit
`..._G099_H1000`.

`RAW_ADAPTER_FIXTURE` tests a reward-free, digest-bound TASK_5 request and
sealed response. It uses fixture scores and is not a Raw-Delta/RKME
implementation. Production Raw remains `NO_GO_RAW_OPERATOR_AUTHORITY` until a
trusted export authority and reward-free membership verifier are implemented;
this release does not expose a production Raw execution path. Live subprocess
execution is intentionally disabled because this companion cannot yet enforce
the old repository and assets as read-only.

```bash
python -m pip install -e '.[test]'
policy-learnware-ope toy --output artifacts/toy --seed 7
policy-learnware-ope real-preflight --output artifacts/real-preflight.json
pytest
```

R0 validates both the source checkout and an untracked `0.4.1b0` wheel built
offline with the declared `setuptools>=77` backend. Installed code never treats
a consumer repository as this companion checkout: its implementation identity
is the installed distribution version plus a digest of sorted package Python
files, and outputs inside any host Git repository are rejected. Source-checkout
identity is read from Git only after the canonical `src/` layout, project name,
and current `cli.py` path are verified.

The real preflight intentionally exits nonzero and records `NO_GO` for the
three presently missing capabilities: verified actor authority, exact
arbitrary-action behavior density, and a per-step oracle bound to
`J(gamma=.99,H=1000)`. Production Raw is separately `NO_GO`. It also exposes
the method-level `NO_GO_OPS_DS_DENSE_HESSIAN_PANEL` and
`NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT` blockers without changing synthetic
toy estimates from `PASS`. No recollection, policy retraining, oracle rewriting,
or old-asset mutation is performed.

Ranking seals are canonical, write-once payloads. Their SHA-256 is retained by
a separate run manifest and must be supplied when loading or joining an oracle.
The CLI summary also emits the run, seal, and oracle-manifest digests so a
caller can retain them outside the artifact directory.
The toy runner refuses a nonempty output directory, preventing a failed rerun
from mixing newly overwritten inputs with an older sealed manifest.
The oracle manifest independently binds context, candidate set, value
convention, and its caller-held digest. Wall-clock measurements are attached
only after the seal and are exported in `runtime.json` and post-join metrics.

`TOY_MVP_PASS` means the method executed a real fit/estimate path on the
synthetic fixture. It does not mean paper-level or official-code parity, and it
does not claim that the production real-asset gates have passed.

The KMIFQE critic is a compact fixed-tanh-feature NumPy adaptation rather than
the official fully trained PyTorch network. The ETM energy is a compact RFF
adaptation rather than the official four-layer MLP. Both export their actual
configuration and remaining drift, and neither claims published benchmark
parity.

Hashes are exact for discrete protocol identity, assets, membership, seeds,
configuration, and sealed ranking artifacts. Floating results are checked with
declared dtype-aware tolerances plus finiteness, mechanism invariants, and
ranking stability; a legitimate backend-rounding difference is not treated as
an asset or protocol mismatch.
