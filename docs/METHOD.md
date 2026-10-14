# Paper-to-code mapping

Reference: Tsouvalas, Saeed, Ozcelebi, *Federated Self-training for Semi-supervised Audio Recognition*, TECS 21(6), 2022, DOI [10.1145/3520128](https://doi.org/10.1145/3520128). The final journal PDF, rather than only the shorter ICASSP version, governs this implementation.

| Paper component | Maintained implementation | Source correspondence / qualification |
|---|---|---|
| §5.2 four spectral/temporal CNN blocks | `fedstar/model.py:build_encoder` | Branch kernels (1,4)/(4,1), widths 24/32/64/128, concatenate, 1×1 convolution, GN 4, ReLU, max pool 2, spatial dropout 0.1, L2 1e-4, global max pool; recovered from `src/model.py`. No attention block. |
| §5.1–5.2 audio | `data.py:read_wave,features` | Mono average and explicit 16kHz resampling; 1-second crops; STFT 400 / hop 160 / FFT 1024, 64 mel60–7800Hz, log magnitude. FFT/band limits and L2 waveform normalization come from source. |
| §5.2 clip evaluation | `train.py:evaluate` | Mean class probabilities over full nonoverlapping one-second segments. Short clips symmetrically padded; trailing partial segment dropped. Original averages logits instead. |
| Algorithm 1 local objective | `method.py:self_training_loss`, `train.py:local_train` | `Ls + beta Lu`, beta 0.5; separate means over labeled/all unlabeled examples, including rejected unlabeled examples in denominator; hard labels from detached softmax(logits/T). Equal-sized L/U batches, cycling L for one pass over U per epoch. |
| §4.3 and Table 2 confidence | `method.py:confidence_threshold` | T=4; threshold 0.9−0.2(1+cos(πr/R)), where r is completed global rounds. Training starts at0.5; 0.9 is reached after the final round. No client-local participation schedule. |
| §5.3 data allocation | `data.py:partition` | Floor class-wise label budgets preserve original class proportions up to integer rounding. Remaining unlabeled data randomly subset by U; independent L/U ownership. Seeded quantity allocation near N/K with configurable variation. Every example assigned once. |
| §5.4 label-skew ablation | `data.py:partition(class_skew=True)` | Restrict labeled class ownership; distribute classes among clients, splitting a class if clients exceed classes. Unlabeled ownership identical with this switch on/off. |
| Algorithm 1 FedAvg | `method.py:aggregate`, `server_app.py:SeededFedAvg` | Weight by exact local retained utterance count Nl+Nu (Nl in supervised mode), not batch count. Uniform random clients without replacement, floor(qK), minimum1. Fresh Adam each local fit, as original source. |
| §4.4 optional centralized pretraining | `pretrain.py`, `model.py:contrastive_loss` | Two stochastic crops per clip; 256-unit projection, learned bilinear matrix, other clips as negatives. Encoder transferred; projection discarded. Original code/checkpoint absent. Explicit reconstruction choice: symmetric cross-view InfoNCE, identity-initialized bilinear matrix, Adam 1e-3, no extra temperature or projection normalization. These details are not specified by the paper. |

## Source discrepancies and repairs

The original public code is preserved under `src/` and at `federatedml-baseline` (`a457290fcd172d5fe8c978e49b2d1bd7df91987c`). The maintained path fixes TensorFlow Addons removal/Keras 3 symbolic API usage and Flower 0.17 signatures. It also corrects concrete problems:

- Original `PSL_Network.train_step` defaults to T=4, but `fit(T=1)` always overrides it. Paper mode uses T=4. `mode: source-loss` exposes the original T=1 and convex `(1-beta)Ls+betaLu` objective. It does not restore every historical preprocessing defect.
- Original training crop is15,690 samples, test frame15,680. Maintained crops are 16,000 per the paper. Tail handling and waveform normalization are explicit above.
- Original single loader maps VoxForge languages to “unknown” and interprets Ambient TSV labels incorrectly. Adapters preserve six language IDs and trim Ambient label whitespace; Speech Commands uses fixed12 labels, including unknown and silence. Train and test share one mapping.
- Original sample-rate metadata is ignored; current adapter resamples. Original per-split `np.unique` label mappings can disagree; current mapping is fixed/shared.
- Original shuffled sample allocation accidentally iterates the pre-shuffle lists, can miscount remainder samples, and truncates U with `len(U)-1`. Current allocation accounts for every retained sample. It is deterministic, but not identical to unrecoverable original client membership.
- Original FedAvg reports `num_batches * batch_size`; actual local utterance counts now weight every update. Empty labeled clients fail with a useful message; a fully supervised dataset is supported without division by zero.

The paper's reported results and original missing experiment ledger do not establish an exact seed, partition, silence-generation rule, or pretrained checkpoint for every figure. This release records choices and validates execution; it does not infer that corrected code reproduces those values.

The sequential runner seeds client IDs directly. Flower seeds selection over sorted node IDs; those node IDs can change between launches, so partial-participation trajectories need not match the sequential engine. Full-participation local updates use the same client seeds in both paths. Flower aborts incomplete cohorts rather than silently changing the averaging denominator.
