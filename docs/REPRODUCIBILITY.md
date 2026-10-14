# Reproducibility record

## Environment and scope

The maintained package is `fedstar/`; `src/` and the old launcher preserve source history. Tested on macOS arm64 CPU with Python 3.11.15, TensorFlow 2.20.0, Keras 3.15.1, NumPy 1.26.4, ml-dtypes 0.5.4, Flower 1.40.0, Ray 2.55.1. The dependency snapshot is `requirements-lock-py311.txt`; it records the validated environment, not a cross-platform wheel lock. Python 3.12 and Linux CI are supported targets but not locally measured environments.

Create the virtualenv in a path without spaces. Ray 2.55.1 worker launch on this macOS host failed when the interpreter/package path contained spaces; symlinking the whole venv did not fix the resolved path. A newly created no-space virtualenv works. No CUDA/GPU execution was validated.

```bash
python3.11 -m venv ~/.venvs/fedstar
source ~/.venvs/fedstar/bin/activate
python -m pip install -r requirements-lock-py311.txt
python -m pip install -e '.[flower,test]'
python -m pip check
python -m pytest -q
fedstar train --config configs/smoke.json
fedstar flower --config configs/smoke.json --output runs/flower-smoke
flwr build
```

The unit suite checks analytic pseudo-label temperature/masking/normalization and gradients, threshold endpoints, exact FedAvg weighting, model gradients/serialization, all three adapters, split overlap/missing data, deterministic partitions and label-skew isolation, resampling, generated WAV training, contrastive gradients and encoder transfer. Historical launcher tests still cover owned-process cleanup and port preservation. The Flower command exercises actual Ray workers and Message API aggregation; it is not a mock or centralized loop.

## Data, partitions and seeds

See [dataset preparation](../data_splits/README.md). JSON manifests retain local absolute audio paths and label mappings; keep them local. Runtime output can also contain those paths. No audio, checkpoint, output, or dataset manifest is bundled. Paths in shipped configs resolve against the checkout; see the native CLI section.

Defaults follow Table 2/main experiments: Adam 1e-3, beta 0.5, T=4, confidence0.5→0.9, one local epoch, 100 rounds, 80% participation, 25% quantity variation. Shipped dataset configs use5 clients and5% labeled data as an explicit starting setting; change them to the figure/table being evaluated. `batch_size=64` is total batch size (32L+32U). Source and paper do not fully specify all rounding and seed choices; see [method mapping](METHOD.md).

Use at least three separately seeded runs for the paper's mean-over-three evaluation, for example2021/2022/2023. These seeds are provided execution choices, not a recovered paper ledger. With a fixed config/seed the local CPU sequential path enables TensorFlow deterministic operations. Hardware/software changes or partial participation in Flower can change trajectories.

## Artifact schema

Each new output directory contains:

- `config.json`: resolved parameters and effective temperature.
- `partitions.json`: class order, ordered records per client (unlabeled targets −1), test IDs. No waveforms; local paths may be present.
- `environment.json`: interpreter, platform, devices, package versions; real-data manifest SHA256 and optional encoder SHA256.
- `metrics.json`: benchmark-reproduction flag (false), runtime, evaluation accuracy/loss by round, participant IDs, exact client weights, local loss, thresholds and accepted pseudo-label counts. Flower includes initial evaluation at round 0.
- `model.keras`: final classifier and encoder; load with `tf.keras.models.load_model`.

Pretraining writes `encoder.keras`, `clips.json` (paths and SHA256), `environment.json`, and `metrics.json` with explicit reconstruction status, seed, epoch/batch settings, clip count and losses. Its default500 epochs with batch size 1024 was not run in release validation. Pretraining inputs must exclude downstream held-out data; use the separate FSD50K corpus as described in §4.4. This release trains that reconstruction from scratch; it ships no pretrained weights.

## Remaining scientific limits

Published benchmarks have not been rerun. Supplied Speech Commands split files refer to 668 generated silence examples absent from the official archives; their original generation recipe is unknown. Original pretraining source/checkpoints, all experiment seeds, exact client memberships, and full result ledgers were not recovered. The maintained preprocessing and objective intentionally correct documented source/paper discrepancies, so passing tests cannot establish numerical equivalence to the paper or historical code. No formal privacy guarantee, secure aggregation, or differential privacy is added by this local training implementation.


## Native Flower CLI

The README's `flwr run . --stream` runs the packaged ClientApp and ServerApp
through Flower 1.40's managed local SuperLink. Both native CLI paths were verified
with two clients and two rounds; client metrics, server metrics and model files
were saved. Generated-data execution is not a dataset benchmark.

`FLWR_HOME` and the local HTTP port in the README isolate this app's runtime from
other projects. Flower keeps the managed SuperLink running after a run. Use
`flwr list`, `flwr log RUN_ID --show`, or `flwr stop RUN_ID` in the same environment.
A stop request stops that run, not the background service. The official
[local runtime guide](https://flower.ai/docs/framework/main/en/how-to-run-flower-locally.html)
explains service lifecycle and logs. If changing virtual environments, inspect
and stop only the matching service process before starting its replacement.
`FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` uses the dependencies already
installed by setup; this avoids a second uncontrolled installation at run time.

The Python environment and its actual package files must live in paths without
spaces; symlinking a space-containing environment is insufficient. The repository
and data paths may contain spaces. Supply absolute `config` and `output` paths in
`--run-config`, as shown in the README. Paths inside files in `configs/` resolve
against that directory's parent (the checkout), even when Flower executes its
bundle elsewhere. For a config outside `configs/`, paths resolve against the
config file's own directory. CLI `--output` overrides resolve in the caller's
working directory. Set `num-supernodes` equal to the configured client count.

Preparation now decodes every input and records SHA-256/byte counts before
atomically publishing its manifest. `fedstar validate --manifest PATH` checks those
hashes again. Training performs structural, label and duplicate-path checks;
run explicit validation after moving or changing data. Preparation and training
refuse to overwrite outputs. Manifests are local artifacts and can reveal paths.
