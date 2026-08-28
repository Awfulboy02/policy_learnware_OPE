# Third-party method provenance

This repository contains clean, NumPy-based project adaptations; it does not
vendor upstream source code, environments, policies, logs, or model weights.

- `FH_FQE_G099_H1000`: finite-horizon project adaptation of Fitted Q
  Evaluation; B03 is the local experimental anchor, not copied code.
- `FH_KMIFQE_G099_H1000`: clean NumPy B20 protocol adaptation informed by
  *Kernel Metric Learning for In-Sample OPE* and the official
  `haanvid/kmifqe` source at commit
  `070f121d29f05638221695690d5b0d1f0e2bf75b` (MIT, Copyright 2024 Haanvid
  Lee). No upstream source is vendored and no official numerical parity is
  claimed.
- `ETM_MBOPE_G099_H1000`: clean NumPy B22 protocol adaptation informed by
  *Offline Transition Modeling via Contrastive Energy Learning* and the
  official `Ruifeng-Chen/Energy-Transition-Models` source at commit
  `2a2c780c0da074b6e7733a3cb6b40b2444452de6` (Apache-2.0). No upstream source
  is vendored; the RFF energy is not the official four-layer MLP and no
  official numerical parity is claimed.
- `DOPE_STYLE_MB_FF_G099_H1000`: project-defined feed-forward model-based
  reference. DOPE supplies benchmark/reporting context, not an algorithm.
- `AR_MBOPE_G099_H1000`: independent method-level reimplementation informed
  by B06; it is not a DOPE algorithm.

Frozen v03 assets remain in their source repository. They are accessed only
through digest-locked, read-only adapters.
