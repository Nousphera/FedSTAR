# FedSTAR

Federated self-training for semi-supervised audio recognition. Each client learns from its labeled audio and confident pseudo-labels, while a server aggregates updates with FedAvg.

[Journal paper](https://doi.org/10.1145/3520128) · [ICASSP paper](https://doi.org/10.1109/ICASSP43922.2022.9746356) · [Project page](https://nousphera.github.io/FedSTAR/) · [Poster](https://nousphera.github.io/FedSTAR/poster.pdf) · [Provenance](PROVENANCE.md)

<img src="images/fedstar.png" alt="FedSTAR training overview" width="60%">

## Install and run

Use Python 3.11 or 3.12. The maintained `fedstar/` package uses TensorFlow 2.20, Keras 3, and Flower 1.40. Put the environment at a path without spaces for Ray workers.

```bash
python3.11 -m venv ~/.venvs/fedstar
source ~/.venvs/fedstar/bin/activate
python -m pip install -e '.[flower,test]'
python -m pytest -q
fedstar train --config configs/smoke.json
fedstar flower --config configs/smoke.json --output runs/flower-smoke
```

Both commands run the four-block audio CNN through two clients and two rounds on generated waveforms; the second uses real Flower ClientApp/ServerApp messages and Ray workers. Outputs include resolved configuration, client partitions, environment, round metrics, and `model.keras`. Existing run directories are never overwritten. Synthetic accuracy does not measure paper performance.

## Prepare data

The maintained adapters support **Speech Commands v2 (12 classes), VoxForge (6 languages), and Ambient Context (10 classes)**. Obtain the audio separately and follow [data preparation](data_splits/README.md). For example:

```bash
fedstar prepare --task voxforge \
  --train-root datasets/voxforge/train --test-root datasets/voxforge/test \
  --train-split data_splits/voxforge/train_split.txt \
  --test-split data_splits/voxforge/test_split.txt --output data/voxforge.json
fedstar train --config configs/voxforge.json --seed 2021
```

Paths inside each split are relative to the corresponding audio root. Preparation checks every file and rejects overlapping train/test paths. Audio is converted to mono and resampled to 16 kHz. Clients receive disjoint utterances; unlabeled targets are masked. The class-skew option restricts **labeled data only**, leaving the unlabeled allocation unchanged.

Paper settings use temperature 4, `Ls + 0.5 Lu`, and a global-round cosine confidence threshold from 0.5 toward 0.9. Set `supervised_only` for the labeled-only baseline. See [method mapping](docs/METHOD.md) for exact preprocessing, loss normalization, aggregation, and documented source differences.

## Optional pretraining

A separate contrastive encoder can initialize FedSTAR:

```bash
fedstar pretrain --clips data/fsd50k-clips.json --output runs/pretrain
```

The JSON file is a list of local FSD50K WAV paths. Then set `pretrained_encoder` to `runs/pretrain/encoder.keras` in the experiment config. This path implements the paper's two-crop, bilinear contrastive design with explicit loss choices. The original pretraining code and checkpoint were not recovered, so this is a **paper-derived reconstruction**, not the historical checkpoint. Defaults are 500 epochs and batch size 1024; use a smaller batch for a bounded check.

## Reproducibility

The release verifies analytic method tests, all three split adapters, generated WAV training, encoder transfer, model serialization, and actual Flower simulation. It does **not** reproduce the published dataset benchmarks. The supplied Speech Commands split needs 668 original generated silence clips whose generation recipe is missing. Read the [reproducibility record](docs/REPRODUCIBILITY.md) for commands, artifacts, seeds, environment, and remaining gaps.

The original `src/`, `run.py`, `config.yml`, and `requirements.txt` remain historical reference code. Install the maintained package above for current training; those old dependencies are not a validated modern environment.

## Citation and license

```bibtex
@article{tsouvalas2022fedstar,
  author = {Tsouvalas, Vasileios and Saeed, Aaqib and Ozcelebi, Tanir},
  title = {Federated Self-training for Semi-supervised Audio Recognition},
  journal = {ACM Transactions on Embedded Computing Systems},
  volume = {21}, number = {6}, year = {2022},
  doi = {10.1145/3520128}
}
```

Code: [Apache-2.0](LICENSE). Dataset licenses and access terms remain with their providers. Original Git history and author attribution are preserved.
