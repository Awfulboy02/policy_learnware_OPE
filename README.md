# Policy Learnware OPE v0.4b

Minimal companion repository for policy-library selection under target-dynamics
shift. It estimates and ranks the same-task frozen candidate policies using a
fixed rewarded behavior log, while keeping the v03 repository, policies,
manifests, logs, and oracle read-only.

The executable MVP includes five finite-horizon estimators, a delegated
`RAW_DELTA_TASK5` comparator, digest-locked actor/asset bridges, ranking seals,
post-seal metrics, and synthetic/known-MDP acceptance tests. The primary value
convention is raw `J(gamma=0.99,H=1000)`.

```bash
python -m pip install -e '.[test]'
policy-learnware-ope toy --output artifacts/toy
pytest
```

Production gates fail closed: sampled `0..63` ordinals are not accepted as
native timesteps, clipped-Gaussian logs without an arbitrary-action exact
density cannot run KMIFQE, stochastic FPO actors require explicit action keys,
and episodic-return-only oracle files cannot be relabeled as discounted values.

`TOY_MVP_PASS` means that each method executes a real fit/estimate path on the
acceptance fixture. It does not claim official-code or paper-architecture
parity, and it is not a real-asset training result; each artifact records the
remaining production-port scope explicitly.
