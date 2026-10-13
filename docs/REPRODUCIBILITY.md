# Reproducibility record

## Scope

The baseline is the original FedSTAR repository at commit `a457290fcd172d5fe8c978e49b2d1bd7df91987c`. See [provenance](../PROVENANCE.md). This maintenance update repairs process startup, argument handling, explicit device selection, and failure reporting. Model layers, losses, pseudo-label thresholds/schedule, data partitioning, seeds, and shipped split files are unchanged.

A passing launcher test does not reproduce a paper result. No complete training run, GPU run, or published accuracy has been validated in this update.

## Environment

Original declarations in `requirements.txt` are TensorFlow 2.3.1, tensorflow-gpu 2.3.2, TensorFlow Addons 0.11.2, standalone Keras 2.3.1, and Flower 0.17.0. They are retained as historical evidence, not an installation recipe. The source uses `tf.keras` and imports NumPy and pandas, which the original file does not declare directly. The launcher additionally needs PyYAML.

Recover a compatible historical Python/platform environment, select one TensorFlow distribution, resolve its NumPy/pandas/Addons constraints, and record a successful import/model/training check before locking versions. The old Flower client signatures and strategy options are not interchangeable with a current Flower release. TensorFlow Datasets is mentioned in the old README but is not imported by this source. No claim of compatibility with current TensorFlow, Keras, or Flower is made.

For the lightweight launcher alone, `pip install -r requirements-launcher.txt` installs the only external dependency. Tests ran on macOS with Python 3.14.4 and PyYAML 6.0.3 on 2026-10-13. The same lightweight suite also passed in an isolated Python 3.12 environment with PyYAML 6.0.3. This host did not have TensorFlow or Flower installed.

## Data and scientific gaps

- The current loader covers the Speech Commands 12-class task. VoxForge and Ambient Context require original dataset-specific loaders; existing raw-loader inspection gives 1/6 train/test classes for VoxForge and 1/697 for Ambient Context. Those outputs are evidence of incompatibility, not valid tasks.
- Speech Commands training references 668 generated silence clips whose construction is not recorded here. The original audio/preprocessing must be recovered before claiming reproduction.
- The pipeline assumes 16 kHz audio and discards the decoded sample-rate value. It does not resample inputs.
- A new/prepared split must keep matching train/test label mappings. The source builds those mappings separately; missing classes can change label IDs.
- Very small labeled partitions, empty clients, or no unlabeled examples can invalidate batching/self-training. Launcher validation rejects obvious empty labeled/unlabeled configurations, but full per-client sufficiency requires inspecting prepared data.
- The baseline weights federated updates using batch-derived example counts. This behavior is preserved, as are the existing confidence schedule, loss normalization, and random-state behavior. These have not been reassessed against the published experiments.
- Pretrained checkpoints and a complete experiment ledger are not supplied. The sample configuration is a short supervised run, not the paper's full experiment matrix.

## Validation

```bash
python -m unittest discover -s tests -v
bash -n run.sh
./run.sh --dry-run
python src/clients.py --help
python src/server.py --help
```

The six launcher/process tests cover configuration-relative paths, intact addresses, CPU/GPU assignment, occupied ports, failed server startup, failed clients, and worker cleanup. Dummy local processes stand in for the training runtime; no audio or remote service is required. The repository Python sources also passed AST parsing.

Before reporting a reproduced result, record the exact commit and resolved dependency versions, prepared audio checksums and silence-generation provenance, configuration, hardware, all random seeds, and round-wise outputs. Verify both supervised and FedSTAR modes using the same historical task definition.
