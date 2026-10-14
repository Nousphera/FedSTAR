# Dataset preparation

The maintained `fedstar prepare` command reads the supplied split files, checks every local audio path, and writes one manifest with a shared class mapping. It never downloads data. Obtain audio under the providers' terms; no audio is bundled here. `src/data.py` is the historical Speech Commands-only loader and is not used by the maintained package.

## Speech Commands v2

Obtain the training and test archives linked by the [TensorFlow dataset documentation](https://www.tensorflow.org/datasets/catalog/speech_commands). Extract to separate roots:

```text
datasets/speech_commands/
├── train/<label>/<file>.wav
└── test/<label>/<file>.wav
```

The class order is `yes,no,up,down,left,right,on,off,stop,go,silence,unknown`. Other words map to unknown. `_silence_`, `silence`, and `training_silence` directories map to silence. Both splits use this fixed order.

**The supplied training split requires 668 generated `_silence_/*.wav` clips** (for example `doing_the_dishes_36.wav`) whose original generation recipe is absent. Official archive extraction alone is insufficient. Obtain those prepared clips to use the historical split; the adapter fails on missing files. Alternatively supply and document a new split and silence preparation method, with no claim that it is the historical dataset.

```bash
fedstar prepare --task speech_commands \
  --train-root datasets/speech_commands/train --test-root datasets/speech_commands/test \
  --train-split data_splits/speech_commands/train_split.txt \
  --test-split data_splits/speech_commands/test_split.txt --output data/speech_commands.json
fedstar train --config configs/speech_commands.json
```

The supplied files list 85,511 train and 4,890 test examples. Preparation checks presence, unique resolved paths, and train/test disjointness. Source files remain unchanged for provenance.

## VoxForge

Arrange the locally obtained [VoxForge](https://www.voxforge.org/) recordings so paths in each split resolve relative to their root. For example, a line `de/Trunk/Audio/Main/16kHz_16bit/.../wav/recording.wav` resolves to `<root>/de/Trunk/Audio/Main/16kHz_16bit/.../wav/recording.wav`. Roots may be the same corpus directory when split paths are disjoint.

The fixed class order is `de,en,es,fr,it,ru`. An unrecognized language fails rather than becoming a keyword “unknown” class.

```bash
fedstar prepare --task voxforge \
  --train-root datasets/voxforge --test-root datasets/voxforge \
  --train-split data_splits/voxforge/train_split.txt \
  --test-split data_splits/voxforge/test_split.txt --output data/voxforge.json
```

The split files identify historical recordings; this release cannot guarantee every upstream recording remains downloadable. No automatic scraper or dataset redistribution is included.

## Ambient Acoustic Contexts

Use the dataset cited as reference 33 in the [paper](https://doi.org/10.1145/3520128). Each supplied row is `relative_filename<TAB>class_name`. Arrange local WAV files beneath the corresponding root. The adapter trims incidental label whitespace and constructs one sorted class order from both splits; the full supplied splits contain10 classes. A subset manifest can have fewer classes for checks and is not the benchmark task.

```bash
fedstar prepare --task ambient_context \
  --train-root datasets/ambient_context --test-root datasets/ambient_context \
  --train-split data_splits/ambient_context/train_split.txt \
  --test-split data_splits/ambient_context/test_split.txt --output data/ambient_context.json
```

All adapters read audio through SoundFile, average channels, resample to 16 kHz with SciPy polyphase filtering, then compute log-mel features. Training uses random one-second crops; evaluation averages probabilities over complete one-second segments. Small clips are padded; incomplete trailing segments are dropped. See [method details](../docs/METHOD.md).
