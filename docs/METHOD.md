# Method and configuration

FedSTAR implements federated audio self-training described in
[Tsouvalas, Saeed and Ozcelebi, ACM TECS 21(6), 2022](https://doi.org/10.1145/3520128).

## Encoder and features

The encoder in `fedstar/model.py` has four spectral/temporal CNN blocks with
widths 24, 32, 64 and 128. Each concatenates `(1,4)` and `(4,1)` branches, then
applies a 1×1 convolution, four-group normalization, ReLU, 2×2 max pooling,
spatial dropout 0.1 and L2 regularization 1e-4. Global max pooling produces the
encoder representation. The classifier is a dense layer over the dataset's
shared class mapping.

Audio channels are averaged and resampled to 16 kHz. Features use one-second
crops, a 400-sample STFT window, hop 160, FFT 1024, 64 mel bins from 60–7800 Hz,
log magnitude and L2 waveform normalization. Evaluation averages class
probabilities over complete nonoverlapping one-second segments. Short clips
are symmetrically padded; a trailing partial segment is dropped.

## Local self-training

The default objective is `Ls + beta * Lu`, with `beta=0.5`. Pseudo-labels are
hard labels from detached `softmax(logits / T)`, with default `T=4`. Labeled and
unlabeled losses are separate means; rejected pseudo-labels contribute zero
while remaining in the unlabeled denominator. Equal-sized labeled/unlabeled
batches cycle the labeled subset during one pass over the unlabeled subset.
`batch_size=64` means 32 labeled plus 32 unlabeled examples.

The confidence threshold follows the global-round schedule
`0.9 - 0.2 * (1 + cos(pi*r/R))`, where `r` is the completed global round count.
Training starts at 0.5 and reaches 0.9 after the final round. Each client fit
creates a fresh Adam optimizer with default learning rate 1e-3.

For a loss ablation, `mode="source-loss"` uses temperature 1 and the convex
objective `(1-beta)*Ls + beta*Lu`. Other preprocessing and sampling settings
remain the same.

## Partitions and aggregation

Class-wise floor budgets preserve label proportions up to integer rounding.
Labeled and unlabeled ownership are assigned independently, with configurable
quantity variation around the mean client size. `U` subsamples the unlabeled
pool. Each retained example belongs to exactly one client. Empty labeled
clients fail before training.

`class_skew=true` restricts labeled class ownership, splitting classes when
there are more clients than classes. It leaves unlabeled ownership unchanged,
so comparisons isolate the labeled-distribution change.

FedAvg weights updates by exact retained utterance counts, `Nl+Nu` (or `Nl` for
supervised-only training). Each round samples `floor(participation * clients)`
clients without replacement, with a minimum of one. Flower rejects incomplete
cohorts. Its runtime node-ID ordering can change between launches, whereas the
sequential engine seeds client IDs directly; preserve participation logs for
comparisons with partial participation.

## Contrastive pretraining

`fedstar pretrain` learns an encoder from separate FSD50K development audio.
Two stochastic crops from each clip pass through the encoder and a 256-unit
projection. Symmetric cross-view InfoNCE uses a learned, identity-initialized
bilinear matrix; other clips in the batch supply negatives. Adam uses 1e-3,
with no additional temperature or projection normalization. These settings
make the bilinear objective in paper §4.4 explicit.

Defaults are 500 epochs and batch size 1024. Save the encoder, discard the
projection and bilinear matrix, and set downstream `pretrained_encoder` to
`encoder.keras`. Supply development clips separately and keep downstream
held-out audio out of pretraining.

## Data and deployment

[Dataset preparation](../data_splits/README.md) defines the Speech Commands
silence recipe, VoxForge language classes and Ambient Context label mapping.
Model updates are exchanged during federation; differential privacy and secure
aggregation require additional mechanisms. Run settings and outputs follow below.

## Data and seeds

Follow [dataset preparation](../data_splits/README.md), then run
`fedstar validate --manifest PATH` to decode files and verify their saved hashes.
Training rechecks structure, labels and duplicate paths. Keep audio available
at the manifest's recorded paths. Manifests and run records can contain local
absolute paths; share them according to the dataset's terms.

Default settings are Adam 1e-3, beta 0.5, temperature 4, confidence 0.5→0.9,
one local epoch, 100 rounds, 80% participation and 25% quantity variation.
Dataset configs start with five clients and 5% labels. `batch_size=64` is a total
batch of 32 labeled plus 32 unlabeled examples. The method definitions above
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

Run the README's `flwr run . --stream` command and match `num-supernodes` to the
configured client count. Pass absolute `config` and `output` paths. The virtual
environment's real path must not contain spaces; repository and data paths may.
Files in `configs/` resolve paths against the checkout; other config files resolve
paths against their own directory. CLI output overrides use the working directory.

Use the same `FLWR_HOME` for `flwr list`, `flwr log RUN_ID --show` and
`flwr stop RUN_ID`. Stopping a run leaves its local SuperLink available. When
changing environments, stop only that environment's service; see the
[Flower runtime guide](https://flower.ai/docs/framework/main/en/how-to-run-flower-locally.html).
The README sets isolated ports and disables automatic runtime dependency installation.

Check the completed run status and saved metrics/model files. `flwr build`
packages the app; the Python `flower` command uses the same apps through Ray,
and `train` executes the shared trainer sequentially. Deployment across machines
requires client-local data, authenticated transport and an evaluation policy.
