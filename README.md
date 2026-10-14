# FedSTAR

Research code for **Federated Self-Training for Semi-Supervised Audio Recognition** by Vasileios Tsouvalas, Aaqib Saeed, and Tanir Ozcelebi. FedSTAR combines federated learning with pseudo-labeling to use unlabeled audio held by clients.

[Journal paper](https://doi.org/10.1145/3520128) · [Preprint](https://arxiv.org/abs/2107.06877) · [ICASSP paper](https://doi.org/10.1109/ICASSP43922.2022.9746356) · [Provenance](PROVENANCE.md)

<img src="images/fedstar.png" alt="FedSTAR training overview" width="60%">

## Status

This repository preserves the original research implementation and adds bounded setup and launcher repairs. The shipped loader implements the **12-class Speech Commands task**. VoxForge and Ambient Context splits are retained as historical artifacts; they cannot be used with the current loader by changing a path alone.

The paper's results have **not been reproduced by this maintenance update**. Read the [reproducibility record](docs/REPRODUCIBILITY.md) before selecting an environment or interpreting results.

## Setup

The training code targets historical TensorFlow 2.3 / TensorFlow Addons and Flower 0.17 APIs. [requirements.txt](requirements.txt) preserves the original declarations; it is not a validated installation lock. In particular, the original file declares two different TensorFlow package versions. See [environment requirements](docs/REPRODUCIBILITY.md#environment) before attempting training; a modern Python installation alone is insufficient.

For launcher inspection and its lightweight tests:

```bash
git clone https://github.com/Nousphera/FedSTAR.git
cd FedSTAR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-launcher.txt
python -m unittest discover -s tests -v
```

Prepare audio using the [dataset instructions](data_splits/README.md). The original generated training silence clips are also required; downloading the archives alone does not produce the complete supplied split.

## Run an experiment

Edit [config.yml](config.yml), then run from any directory:

```bash
./run.sh --dry-run
./run.sh
```

Use the script's full path when outside the checkout. `PYTHON=/path/to/python ./run.sh` selects the prepared training interpreter. Alternatively, run `python run.py --config /path/to/config.yml`. Relative `DATA_DIR` values resolve against the configuration file's directory.

The default configuration uses five CPU clients, two participants per round, two rounds, one local epoch, 5% labeled data, and the supervised baseline (`FEDSTAR: false`). Set `FEDSTAR: true` to enable self-training. Keep both labeled and unlabeled examples in that mode.

For NVIDIA GPUs, set `server.GPU_ID` to a device ID and `clients.GPU_IDS` to a comma-separated list, for example `"0,1"`. Clients use those IDs round-robin; `GPU_MEMORY_SIZE` limits each TensorFlow process. Empty IDs select CPU. GPU memory is not reserved globally, so account for concurrent clients and the server yourself. Requested GPUs that TensorFlow cannot see cause an error.

The launcher waits for the local server, refuses an occupied address, propagates worker failures, and cleans up only the processes it starts. Its startup timeout defaults to 120 seconds and can be changed with `--startup-timeout`. Full training still requires a compatible environment and complete audio data.

## Citation

```bibtex
@article{tsouvalas2022fedstar,
  author = {Tsouvalas, Vasileios and Saeed, Aaqib and Ozcelebi, Tanir},
  title = {Federated Self-training for Semi-supervised Audio Recognition},
  journal = {ACM Transactions on Embedded Computing Systems},
  volume = {21},
  number = {6},
  year = {2022},
  doi = {10.1145/3520128}
}

@inproceedings{tsouvalas2022dataefficient,
  author = {Tsouvalas, Vasileios and Saeed, Aaqib and Ozcelebi, Tanir},
  title = {Federated Self-Training for Data-Efficient Audio Recognition},
  booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages = {476--480},
  year = {2022},
  doi = {10.1109/ICASSP43922.2022.9746356}
}
```

Paper metadata was checked against Crossref on 2026-10-13. Software is distributed under [Apache-2.0](LICENSE); dataset licenses and terms remain with their respective providers.
<!-- nousphera-project-resources -->
## Project resources

[Project page](https://nousphera.github.io/FedSTAR/) · [Research overview](https://nousphera.github.io/projects/fedstar/) · [A0 poster](https://nousphera.github.io/FedSTAR/poster.pdf) · [Paper](https://doi.org/10.1145/3520128)

This repository contains the research implementation. The project page and poster summarize the published method and reported experiments; this documentation update does not claim an independent reproduction of those experiments.
