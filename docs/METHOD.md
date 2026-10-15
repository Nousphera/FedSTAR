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
The [experiment guide](REPRODUCIBILITY.md) covers saved artifacts, seeds and
native Flower execution. Model updates are exchanged during federation;
differential privacy and secure aggregation require additional mechanisms.
[Source credits](../PROVENANCE.md) identify the code and license.
