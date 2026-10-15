"""Checked audio manifests shared by PyTorch training and Flower workers."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

WORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
CLASSES = {
    "speech_commands": list(WORDS) + ["silence", "unknown"],
    "ambient_context": ["Applause", "Crowd", "Door", "Laughter", "Others", "Silence",
                        "Television", "TypingClicking", "Walk"],
    "voxforge": ["de", "en", "es", "fr", "it", "ru"],
    "fsd50k": [],
}
DATASETS = ("speech_commands", "ambient_context", "voxforge")


@dataclass
class DataBundle:
    train: list
    test: list
    classes: list
    fingerprint: str


def checksum(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_fingerprint(value):
    payload = {k: v for k, v in value.items() if k != "fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def dataset_directory(dataset, root):
    if dataset not in CLASSES:
        raise ValueError(f"Unknown dataset: {dataset}; choose {', '.join(CLASSES)}")
    root = Path(root).expanduser().resolve()
    return root if (root / "manifest.json").is_file() else root / dataset


def check_records(dataset, records, *, verify=False, populate=False):
    """Validate ownership, labels, files and (on preparation/verification) audio bytes."""
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset requires nonempty records")
    classes = CLASSES[dataset]
    ids, files, inodes, hashes = set(), set(), set(), {}
    output, splits = [], set()
    for original in records:
        row = dict(original)
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("Missing or duplicate example ID")
        ids.add(identifier)
        split = row.get("split")
        if split not in (("train",) if dataset == "fsd50k" else ("train", "test")):
            raise ValueError("Invalid dataset split")
        splits.add(split)
        label = row.get("label")
        if type(label) is not int or (label != -1 if dataset == "fsd50k" else not 0 <= label < len(classes)):
            raise ValueError("Invalid class label")
        path = Path(row["path"]).expanduser()
        if not path.is_absolute():
            raise ValueError("Manifest audio paths must be absolute")
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {dataset} audio: {path}")
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if path in files or inode in inodes:
            raise ValueError(f"Duplicate file or train/test leakage: {path}")
        files.add(path)
        inodes.add(inode)
        row["path"] = str(path)
        if not populate and (row.get("bytes") != stat.st_size or not isinstance(row.get("sha256"), str)):
            raise ValueError(f"Audio file changed or has no integrity metadata: {path}")
        if populate or verify:
            import numpy as np
            import soundfile as sf
            # Inspect all samples, including long clips, without retaining a corpus in RAM.
            with sf.SoundFile(path) as audio:
                if audio.frames < 1 or audio.samplerate < 1 or audio.channels < 1:
                    raise ValueError(f"Empty/invalid audio: {path}")
                for block in audio.blocks(blocksize=65536, dtype="float32"):
                    if not np.isfinite(block).all():
                        raise ValueError(f"Nonfinite audio: {path}")
            current = checksum(path)
            if not populate and row["sha256"] != current:
                raise ValueError(f"Audio checksum changed: {path}")
            row.update(sha256=current, bytes=stat.st_size)
        digest = row["sha256"]
        if digest in hashes and hashes[digest] != split:
            raise ValueError(f"Train/test leakage: identical audio bytes at {path}")
        hashes[digest] = split
        output.append(row)
    required = {"train"} if dataset == "fsd50k" else {"train", "test"}
    if splits != required:
        raise ValueError(f"Every required split must be nonempty: {sorted(required)}")
    return output


def _load(dataset, root, verify):
    directory = dataset_directory(dataset, root)
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{dataset} is not prepared at {directory}. "
                                f"Run python scripts/prepare_data.py --dataset {dataset} --data-root {root}")
    value = json.loads(path.read_text())
    if value.get("schema") != 2 or value.get("dataset") != dataset:
        raise ValueError(f"Wrong dataset manifest: {path}")
    if value.get("classes") != CLASSES[dataset]:
        raise ValueError("Manifest class order differs from the dataset class mapping")
    if value.get("fingerprint") != manifest_fingerprint(value):
        raise ValueError(f"Manifest fingerprint changed; metadata was modified: {path}")
    records = check_records(dataset, value["records"], verify=verify)
    if verify:
        for asset in value.get("metadata_files", []):
            source = Path(asset["path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            if checksum(source) != asset["sha256"]:
                raise ValueError(f"Source metadata checksum changed: {source}")
    return DataBundle(
        [{k: r[k] for k in ("id", "path", "label")} for r in records if r["split"] == "train"],
        [{k: r[k] for k in ("id", "path", "label")} for r in records if r["split"] == "test"],
        list(value["classes"]), value["fingerprint"])


def load_data(dataset, root="data"):
    """Load a prepared dataset; validate labels, manifest, ownership and file sizes."""
    return _load(dataset, root, False)


def verify_dataset(dataset, root="data"):
    """Decode and hash every file before starting a training run."""
    return _load(dataset, root, True)
