# Dataset preparation

The current `src/data.py` loader implements the Speech Commands task: ten command words, silence, and unknown. It reads one relative WAV path per line, infers labels from the first directory component, and expects:

```text
speech_commands/
├── Data/
│   ├── Train/<label>/<file>.wav
│   └── Test/<label>/<file>.wav
├── train_split.txt
└── test_split.txt
```

## Speech Commands v2

From the repository root, download and extract the official archives:

```bash
mkdir -p datasets/speech_commands/Data/Train datasets/speech_commands/Data/Test
curl --fail --location http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz \
  --output datasets/speech_commands/train.tar.gz
curl --fail --location http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz \
  --output datasets/speech_commands/test.tar.gz
tar -xzf datasets/speech_commands/train.tar.gz -C datasets/speech_commands/Data/Train
tar -xzf datasets/speech_commands/test.tar.gz -C datasets/speech_commands/Data/Test
cp data_splits/speech_commands/train_split.txt datasets/speech_commands/
cp data_splits/speech_commands/test_split.txt datasets/speech_commands/
```

Source: [TensorFlow Speech Commands dataset documentation](https://www.tensorflow.org/datasets/catalog/speech_commands). Audio is not bundled in this repository.

**Additional original data are required.** The supplied training split contains 668 `_silence_/*.wav` entries (for example `doing_the_dishes_36.wav`). Their generation procedure and generated clips are not included. Obtain the original prepared clips and place them under `Data/Train/_silence_/` to use the historical split. Do not silently drop them or invent a segmentation rule and call the resulting experiment a reproduction.

Check every referenced file before starting training:

```bash
python - <<'PYTHON'
from pathlib import Path
root = Path('datasets/speech_commands')
missing = []
for split, folder in [('train', 'Train'), ('test', 'Test')]:
    entries = (root / f'{split}_split.txt').read_text().splitlines()
    missing.extend(root / 'Data' / folder / entry for entry in entries
                   if not (root / 'Data' / folder / entry).is_file())
print(f'Missing audio files: {len(missing)}')
for path in missing[:10]:
    print(path)
raise SystemExit(bool(missing))
PYTHON
```

This checks file presence only. The feature pipeline assumes 16 kHz audio and does not resample inputs; sample-rate and waveform checks are still needed for externally prepared data. The included splits contain 85,511 training and 4,890 test entries.

## Historical splits

[VoxForge](voxforge) and [Ambient Context](ambient_context) splits are retained unchanged for provenance. They are not supported by the shipped Speech Commands loader:

- VoxForge has six language labels. The current training loader would collapse them into the unknown class.
- Ambient Context uses tab-separated filenames and labels, whereas the current loader reads path-only rows.

Dataset-specific preparation and label mapping must be recovered and validated before using these splits. Merely changing `DATA_DIR` does not reproduce either experiment. See [the reproducibility record](../docs/REPRODUCIBILITY.md).
