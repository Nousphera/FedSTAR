# Dataset preparation

The `fedstar prepare` command reads the supplied split files, checks every local audio path, and writes one manifest with a shared class mapping. It never downloads data. Obtain audio under the providers' terms; no audio is bundled here.

## Speech Commands v2

The [official TensorFlow dataset builder](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/speech_commands/speech_commands_dataset_builder.py)
identifies the training and test archives. The downloader uses their
Google Storage HTTPS endpoints, stages extraction in a temporary directory,
rejects traversal/links/special files, records archive SHA-256 values and renames
the complete result into a new destination. It never replaces existing data.

```bash
fedstar download-speech --output datasets/speech_commands
fedstar speech-commands --train-root datasets/speech_commands/train \
  --test-root datasets/speech_commands/test --output data/speech_commands --seed 2021
fedstar validate --manifest data/speech_commands/manifest.json
```

Skip the first command when those archives are already extracted. The preparation
uses sorted WAV paths, excludes `validation_list.txt` and `testing_list.txt` from
training, and uses the official test archive. All non-command words map to
unknown; all ten command words retain their names. Class order is
`yes,no,up,down,left,right,on,off,stop,go,silence,unknown`.

A seeded NumPy generator chooses a background-noise recording uniformly, then
an inclusive valid start offset for each of 668 one-second training silence clips.
No amplitude scaling is applied; clips are saved as mono 16 kHz PCM16. Use
`--silence-count` to change the count. The manifest records seed, each noise-file
checksum, source, crop offset and all example hashes. Input recordings stay in
place; generated noise clips live in `data/speech_commands/silence/` alongside
`manifest.json`, `train.txt` and `test.txt`. Keep that directory and the input
corpus together. The provided Speech Commands config points at this manifest.

The supplied `train_split.txt`/`test_split.txt` list 85,511 training and 4,890
test examples. They require their complete audio set, including 668 generated
training silence files. To use those lists, provide every referenced recording:

```bash
fedstar prepare --task speech_commands \
  --train-root /path/to/train --test-root /path/to/test \
  --train-split data_splits/speech_commands/train_split.txt \
  --test-split data_splits/speech_commands/test_split.txt --output data/supplied-splits.json
```

Set `manifest` in a copied config to the resulting file. Missing files fail
explicitly. The official-heldout recipe above creates its own silence clips
and records its distinct split recipe in the manifest.

## VoxForge

Arrange the locally obtained [VoxForge](https://www.voxforge.org/) recordings so paths in each split resolve relative to their root. For example, a line `de/Trunk/Audio/Main/16kHz_16bit/.../wav/recording.wav` resolves to `<root>/de/Trunk/Audio/Main/16kHz_16bit/.../wav/recording.wav`. Roots may be the same corpus directory when split paths are disjoint.

The fixed class order is `de,en,es,fr,it,ru`. An unrecognized language fails rather than becoming a keyword “unknown” class.

```bash
fedstar prepare --task voxforge \
  --train-root datasets/voxforge --test-root datasets/voxforge \
  --train-split data_splits/voxforge/train_split.txt \
  --test-split data_splits/voxforge/test_split.txt --output data/voxforge.json
```

Acquire the recordings referenced by the split lists from their providers.
Preparation requires every referenced file to be present locally.

## Ambient Acoustic Contexts

Use the dataset cited as reference 33 in the [paper](https://doi.org/10.1145/3520128). Each supplied row is `relative_filename<TAB>class_name`. Arrange local WAV files beneath the corresponding root. The adapter trims incidental label whitespace and constructs one sorted class order from both splits; the full supplied splits contain 10 classes. Subset manifests may use fewer classes for small experiments; the manifest records their class mapping.

```bash
fedstar prepare --task ambient_context \
  --train-root datasets/ambient_context --test-root datasets/ambient_context \
  --train-split data_splits/ambient_context/train_split.txt \
  --test-split data_splits/ambient_context/test_split.txt --output data/ambient_context.json
```

All adapters read audio through SoundFile, average channels, resample to 16 kHz with SciPy polyphase filtering, then compute log-mel features. Training uses random one-second crops; evaluation averages probabilities over complete one-second segments. Small clips are padded; incomplete trailing segments are dropped. See [method details](../docs/METHOD.md).


## FSD50K pretraining inputs

Download and extract the development audio using the instructions and checksum
list on the [authors' FSD50K record](https://zenodo.org/records/4060432).
The distribution uses multipart ZIP archives; follow its merge instructions
before extraction. Keep the metadata and per-clip license information. This
project does not download that roughly 25 GB distribution or grant dataset rights.

```bash
fedstar prepare-pretraining --root /path/to/FSD50K.dev_audio --output data/fsd50k-clips.json
fedstar pretrain --clips data/fsd50k-clips.json --epochs 1 --batch-size 2 --output runs/pretrain-check
```

The preparation command validates every local WAV, rejects duplicate/escaping
resolved paths and atomically writes the sorted list consumed by pretraining.
The pretrainer records input checksums in `clips.json`. Use development audio
only; do not add downstream held-out clips. See the [experiment guide](../docs/METHOD.md) for pretraining defaults
and encoder transfer.

## Validate any classification manifest

Run `fedstar validate --manifest data/voxforge.json` (or the corresponding
Ambient/Speech Commands manifest) before training. The check decodes each file,
checks saved hashes, verifies labels and rejects duplicate/overlapping paths.
The training entry point repeats structural and label checks without rehashing
the entire corpus for each client.
