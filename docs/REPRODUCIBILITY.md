# Experiment guide

## Environment

Use Python 3.11 or 3.12 and the [setup script](../scripts/setup_env.sh).
`requirements-lock-py311.txt` records a Python 3.11.15/macOS arm64 CPU environment
with TensorFlow 2.20.0, Keras 3.15.1, NumPy 1.26.4, ml-dtypes 0.5.4,
Flower 1.40.0 and Ray 2.55.1. Run `python -m pip check` and
`python -m pytest -q` after installation.

Tests cover pseudo-label temperature/masking/gradients, threshold endpoints,
exact aggregation, model updates/serialization, all dataset adapters,
deterministic partitions, label-skew isolation, resampling, generated-WAV
training, contrastive gradients, encoder transfer and Flower message execution.

## Data and seeds

Follow [dataset preparation](../data_splits/README.md), then run
`fedstar validate --manifest PATH` to decode files and verify their saved hashes.
Training rechecks structure, labels and duplicate paths. Keep audio available
at the manifest's recorded paths. Manifests and run records can contain local
absolute paths; share them according to the dataset's terms.

Default settings are Adam 1e-3, beta 0.5, temperature 4, confidence 0.5→0.9,
one local epoch, 100 rounds, 80% participation and 25% quantity variation.
Dataset configs start with five clients and 5% labels. `batch_size=64` is a total
batch of 32 labeled plus 32 unlabeled examples. The [method guide](METHOD.md)
defines label rounding, class-skew allocation and the optional loss mode.

Use at least three separately seeded runs for mean-over-three evaluation, for
example seeds 2021, 2022 and 2023. Record every seed and choose distinct output
directories. The sequential CPU path enables TensorFlow deterministic operations.
Hardware/software changes or Flower runtime node-ID assignment can change
partial-participation trajectories; preserve participant and sample-count logs.

## Outputs and encoder transfer

| File | Contents |
|---|---|
| `config.json` | Resolved parameters and effective temperature |
| `partitions.json` | Shared class order, client records/label masks and test IDs |
| `environment.json` | Interpreter/platform/devices/packages, manifest and optional encoder hashes |
| `metrics.json` | Dataset/mode, runtime, round accuracy/loss, participants, exact weights, local losses, thresholds and accepted pseudo-label counts |
| `model.keras` | Final classifier and encoder; load with `tf.keras.models.load_model` |

Flower includes initial evaluation at round 0. Existing output directories are
rejected so each experiment retains its own files.

Pretraining writes `encoder.keras`, `clips.json` (paths and SHA-256),
`environment.json` and `metrics.json` with the objective, seed, epochs, batch
size, learning rate, clip count and losses. Defaults are 500 epochs and batch
size 1024. Use separate FSD50K development audio and exclude downstream held-out
clips. Set `pretrained_encoder` in the downstream config to the saved encoder;
otherwise training starts from random initialization.

## Native Flower CLI

The README's `flwr run . --stream` starts the packaged ClientApp and ServerApp
through Flower 1.40's local SuperLink. Set `num-supernodes` to the configured
client count. Use absolute `config` and `output` paths in `--run-config`.

Create the virtual environment in a path without spaces, including the actual
package location; a symlink to a space-containing environment is insufficient
for Ray workers. Repository and dataset paths may contain spaces. Paths inside
files in `configs/` resolve against the checkout. Other config files resolve
paths against their own directory; CLI output overrides use the caller's
working directory.

`FLWR_HOME`, `FLWR_LOCAL_SUPERLINK_HTTP_API_PORT` and
`FLWR_LOCAL_CONTROL_API_PORT` isolate the app's local runtime. The README sets
`FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` to use the setup environment.
Flower keeps its SuperLink service running between jobs. Use `flwr list`,
`flwr log RUN_ID --show` and `flwr stop RUN_ID` with the same environment.
Stopping a run leaves the service available for future runs. Consult the
[local runtime guide](https://flower.ai/docs/framework/main/en/how-to-run-flower-locally.html)
for service lifecycle and logs; when switching environments, stop only the
service associated with that `FLWR_HOME`.

Confirm the run reports completion and inspect the saved metrics/model files.
`flwr build` packages the app. The Python `flower` subcommand runs the same apps
through Ray, and `train` runs the shared trainer sequentially for debugging.
Deployment across machines additionally requires client-local data provisioning,
authenticated transport and a server-side evaluation policy.
