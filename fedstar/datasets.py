"""Documented dataset import recipes; never run automatically at import time."""
from pathlib import Path, PurePosixPath
import os
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

import numpy as np
import soundfile as sf
from .data import CLASSES, read_splits, read_wave
from .preparation import check_records, checksum, write_json

SPEECH_URLS = {
    "train": "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
    "test": "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz",
}


def extract_tar(archive, destination):
    """Reject links, traversal and special files before extracting any member."""
    destination = Path(destination)
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        seen = set()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if name in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            seen.add(name)
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def download_speech(output):
    """Fetch the official archives into a new directory, with provenance hashes."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".speech-download-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        provenance = {}
        for split, url in SPEECH_URLS.items():
            archive = Path(temporary) / f"{split}.tar.gz"
            with urlopen(url, timeout=60) as response, archive.open("xb") as stream:
                shutil.copyfileobj(response, stream)
            provenance[split] = {"url": url, "sha256": checksum(archive)}
            extract_tar(archive, stage / split)
            archive.unlink()
        write_json(stage / "download.json", provenance)
        stage.rename(output)


def prepare_speech(train_root, test_root, output, seed=2021, silence_count=668):
    """Official train/test plus a recorded replacement silence recipe."""
    if seed < 0 or silence_count < 1:
        raise ValueError("seed must be nonnegative and silence-count positive")
    train_root, test_root = Path(train_root).resolve(), Path(test_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    # Use official validation/testing lists to avoid training on held-out speakers.
    excluded = set()
    for name in ("validation_list.txt", "testing_list.txt"):
        excluded.update((train_root / name).read_text().splitlines())
    train = [p.relative_to(train_root).as_posix() for p in sorted(train_root.glob("*/*.wav"))
             if p.parent.name != "_background_noise_" and p.relative_to(train_root).as_posix() not in excluded]
    test = [p.relative_to(test_root).as_posix() for p in sorted(test_root.glob("*/*.wav"))]
    noises = sorted((train_root / "_background_noise_").glob("*.wav"))
    if not noises or not train or not test:
        raise ValueError("Expected official training/test archives and background noise WAVs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".speech-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        (stage / "train.txt").write_text("\n".join(train) + "\n")
        (stage / "test.txt").write_text("\n".join(test) + "\n")
        rows, classes = read_splits("speech_commands", train_root, test_root, stage / "train.txt", stage / "test.txt")
        rows = check_records(rows, classes, ("train", "test"), decode=read_wave)
        # Reject copied train/test files even when their roots differ.
        train_hashes = {r["sha256"] for r in rows if r["split"] == "train"}
        if train_hashes & {r["sha256"] for r in rows if r["split"] == "test"}:
            raise ValueError("Identical audio bytes occur in train and test")
        (stage / "silence").mkdir()
        rng = np.random.default_rng(seed)
        noise_arrays = [read_wave(p) for p in noises]
        recipe = []
        for i in range(silence_count):
            index = int(rng.integers(len(noises)))
            wave = noise_arrays[index]
            if len(wave) < 16000:
                raise ValueError("Background noise must be at least one second")
            start = int(rng.integers(len(wave) - 16000 + 1))
            name = f"silence/{i:06d}.wav"
            sf.write(stage / name, wave[start:start + 16000], 16000, subtype="PCM_16")
            rows.append({"id": "train/" + name, "path": str(output / name), "class": "silence",
                         "label": classes.index("silence"), "split": "train", "sha256": checksum(stage / name),
                         "bytes": (stage / name).stat().st_size})
            recipe.append({"file": name, "source": str(noises[index]), "start_16khz": start})
        write_json(stage / "manifest.json", {"schema": 1, "task": "speech_commands", "classes": classes,
                   "records": rows, "preparation": {"recipe": "official-heldout-plus-seeded-noise-v1", "seed": seed,
                   "historical_paper_split": False, "silence_count": silence_count,
                   "noise_sources": [{"path": str(p), "sha256": checksum(p)} for p in noises], "silence": recipe}})
        stage.rename(output)
    return output / "manifest.json"
