# Provenance

This repository is an independent Nousphera copy of [FedSTAR](https://github.com/FederatedML/FedSTAR). The original FederatedML repository remains in place.

- Source commit: [`a457290fcd172d5fe8c978e49b2d1bd7df91987c`](https://github.com/FederatedML/FedSTAR/commit/a457290fcd172d5fe8c978e49b2d1bd7df91987c)
- Source branch: `main`
- Baseline tag in this repository: `federatedml-baseline`
- Original Git history and author attribution are retained.

Maintenance changes after this baseline improve documentation, setup, and validation. They do not constitute a new reproduction of the published experiments. For the original code and configuration, inspect the baseline tag.

GitHub issues, pull requests, stars, releases, and Pages configuration are not part of Git history and have not been transferred. Existing discussions remain at the source repository.

## Maintained training release (2026-10-15)

The `fedstar/` implementation ports the public source above to TensorFlow 2.20/Keras 3 and Flower 1.40, repairs data adapters and exact aggregation counts, and uses the final journal paper's defaults. Source behavior that differs from the paper is recorded in [METHOD.md](docs/METHOD.md). The original `src/`, old requirements, launcher, and split lists remain available unchanged by this release. Generic simulator/Flower orchestration follows the same maintained pattern as the author's FedSER release; FedSTAR's architecture, temperature, confidence schedule and data semantics are distinct.

The optional `fedstar/pretrain.py` is a paper-derived reconstruction: original pretraining source and weights were not available in this repository. Its bilinear loss choices are documented and must not be attributed to a recovered historical implementation. Original source authors and Apache-2.0 terms are retained. New maintained code is distributed under the same repository license.
