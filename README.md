# FedSTAR

**Federated Self-training for Semi-supervised Audio Recognition**

Vasileios Tsouvalas · Aaqib Saeed · Tanir Ozcelebi · ACM TECS 2022

[Paper](https://doi.org/10.1145/3520128) · [Project](https://nousphera.github.io/FedSTAR/) · [Poster](https://nousphera.github.io/FedSTAR/poster.pdf)

## Abstract

Federated Learning is a distributed machine learning paradigm dealing with decentralized and personal datasets. Since data reside on devices such as smartphones and virtual assistants, labeling is entrusted to the clients or labels are extracted in an automated way. Specifically, in the case of audio data, acquiring semantic annotations can be prohibitively expensive and time-consuming. As a result, an abundance of audio data remains unlabeled and unexploited on users’ devices. Most existing federated learning approaches focus on supervised learning without harnessing the unlabeled data. In this work, we study the problem of semi-supervised learning of audio models via self-training in conjunction with federated learning. We propose FedSTAR to exploit large-scale on-device unlabeled data to improve the generalization of audio recognition models. We further demonstrate that self-supervised pre-trained models can accelerate the training of on-device models, significantly improving convergence within fewer training rounds. We conduct experiments on diverse public audio classification datasets and investigate the performance of our models under varying percentages of labeled and unlabeled data. Notably, we show that with as little as 3% labeled data available, FedSTAR on average can improve the recognition rate by 13.28% compared to the fully supervised federated model.

## Setup

Use Python 3.11 (tested on macOS arm64 CPU); Python 3.12 is supported but untested.
Create the environment at a path without spaces for Flower/Ray workers.

```bash
git clone https://github.com/Nousphera/FedSTAR.git
cd FedSTAR
bash scripts/setup_env.sh "$HOME/.venvs/fedstar"
source "$HOME/.venvs/fedstar/bin/activate"
python -m pytest -q
```

The script creates a new environment, installs `.[flower,test]` constrained by
`requirements-lock-py311.txt`, and checks dependencies. It never changes an
existing environment. The maintained package is `fedstar/`; historical `src/`
and `requirements.txt` are documented in [provenance](PROVENANCE.md).

## Datasets

Speech Commands v2 can be downloaded from the official TensorFlow archive. The
command downloads roughly 2.5 GB and needs additional extraction space:

```bash
python scripts/prepare_data.py download-speech --output datasets/speech_commands
python scripts/prepare_data.py speech-commands \
  --train-root datasets/speech_commands/train --test-root datasets/speech_commands/test \
  --output data/speech_commands --seed 2021
python scripts/prepare_data.py validate --manifest data/speech_commands/manifest.json
```

If already downloaded, start with `speech-commands`. This recipe excludes official
validation/test utterances from training and creates 668 seeded one-second
background-noise clips. It records all sample hashes and crop offsets. It is an
explicit replacement for the unrecovered historical silence-generation recipe,
not the paper's exact split.

[VoxForge and Ambient Context import commands](data_splits/README.md) use supplied
split files and locally acquired audio. All three adapters check split ownership,
decode every recording, and write one shared class mapping with content hashes.
They convert audio to mono and 16 kHz during loading. Preparation never overwrites
existing output. See `python scripts/prepare_data.py --help` for every entry point.

## Run

Run the native Flower ClientApp/ServerApp. The smoke uses two clients, two rounds
and generated waveforms with the full four-block CNN:

```bash
export FLWR_HOME="$HOME/.cache/fedstar-flower"
export FLWR_LOCAL_SUPERLINK_HTTP_API_PORT=39132
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
flwr run . --stream \
  --run-config "config='$(pwd)/configs/smoke.json' output='$(pwd)/runs/flower-smoke'" \
  --federation-config 'num-supernodes=2 client-resources-num-cpus=1 init-args-num-cpus=2'
```

After preparing Speech Commands, use its five-client configuration:

```bash
flwr run . --stream \
  --run-config "config='$(pwd)/configs/speech_commands.json' output='$(pwd)/runs/speech-commands-flower'" \
  --federation-config 'num-supernodes=5 client-resources-num-cpus=1 init-args-num-cpus=2'
```

For the other datasets, substitute `configs/voxforge.json` or
`configs/ambient_context.json`. Match `num-supernodes` to `clients` in the config.
Sequential debugging uses `fedstar train --config configs/smoke.json --output runs/sequential-smoke`.
See [Flower lifecycle and path rules](docs/REPRODUCIBILITY.md#native-flower-cli).

Optional contrastive pretraining uses separately acquired
[FSD50K development audio](https://zenodo.org/records/4060432):

```bash
python scripts/prepare_data.py prepare-pretraining --root /path/to/FSD50K.dev_audio \
  --output data/fsd50k-clips.json
fedstar pretrain --clips data/fsd50k-clips.json --output runs/pretrain
```

Set `pretrained_encoder` in the downstream config to the resulting
`runs/pretrain/encoder.keras`. This is a paper-derived reconstruction; original
pretraining code and weights were not recovered. The default 500 epochs and
batch size 1024 require substantial resources; use `--epochs 1 --batch-size 2`
for a bounded compatibility check.

## Outputs and reproducibility

Each new run saves resolved configuration, disjoint client partitions and label
masks, environment and input-manifest hashes, round/client metrics, and
`model.keras`. Choose a fresh output directory for every seed and configuration.
The defaults implement `Ls + 0.5 Lu`, temperature 4, and the paper's global-round
confidence schedule. [Method details](docs/METHOD.md) and
[validation/provenance](docs/REPRODUCIBILITY.md) explain implementation choices.
Generated-data execution does not establish published benchmark accuracy;
full-corpus, GPU and historical-checkpoint reproduction remain unverified.

## Citation

```bibtex
@article{tsouvalas2022fedstar,
  author = {Tsouvalas, Vasileios and Saeed, Aaqib and Ozcelebi, Tanir},
  title = {Federated Self-training for Semi-supervised Audio Recognition},
  journal = {ACM Transactions on Embedded Computing Systems},
  year = {2022},
  volume = {21},
  number = {6},
  articleno = {74},
  numpages = {26},
  doi = {10.1145/3520128},
  url = {https://doi.org/10.1145/3520128}
}
```

## License

Code: [Apache-2.0](LICENSE). Dataset, publication and external-model licenses remain with their providers.
