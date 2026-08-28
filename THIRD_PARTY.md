# Third-party method provenance

This repository contains clean, NumPy-based project adaptations; it does not
vendor upstream source code, environments, policies, logs, or model weights.

- `FH_FQE_G099_H1000`: finite-horizon project adaptation of Fitted Q
  Evaluation; B03 is the local experimental anchor, not copied code.
- `FH_KMIFQE_G099_H1000`: finite-horizon project adaptation informed by B20
  (Kernel Metric Learning for In-Sample OPE); no official-code claim.
- `ETM_MBOPE_G099_H1000`: compact contrastive-energy feasibility
  implementation informed by B22; not an official ETM port.
- `DOPE_STYLE_MB_FF_G099_H1000`: project-defined feed-forward model-based
  reference. DOPE supplies benchmark/reporting context, not an algorithm.
- `AR_MBOPE_G099_H1000`: independent method-level reimplementation informed
  by B06; it is not a DOPE algorithm.

Frozen v03 assets remain in their source repository. They are accessed only
through digest-locked, read-only adapters.
