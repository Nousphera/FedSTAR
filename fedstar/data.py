"""Explicit split adapters and deterministic disjoint client ownership."""

from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import tensorflow as tf

WORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
CLASSES = {
    "speech_commands": list(WORDS) + ["silence", "unknown"],
    "voxforge": ["de", "en", "es", "fr", "it", "ru"],
}


def read_splits(task, train_root, test_root, train_split, test_split):
    if task not in {*CLASSES, "ambient_context"}:
        raise ValueError("Unknown task")
    parsed = []
    seen = set()
    for split, root, manifest in [
        ("train", Path(train_root), train_split),
        ("test", Path(test_root), test_split),
    ]:
        for line in Path(manifest).read_text().splitlines():
            if not line.strip():
                continue
            if task == "ambient_context":
                name, label = line.split("\t")
                label = label.strip()
            else:
                name = line.strip()
                label = Path(name).parts[0]
                if task == "speech_commands":
                    label = (
                        "silence"
                        if label in {"training_silence", "_silence_", "silence"}
                        else label
                    )
                    label = label if label in CLASSES[task] else "unknown"
                elif label not in CLASSES[task]:
                    raise ValueError(f"Unknown VoxForge language: {label}")
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Split paths must stay inside audio root")
            if path in seen:
                raise ValueError(f"Duplicate or overlapping train/test audio: {path}")
            if not path.is_file():
                raise FileNotFoundError(path)
            seen.add(path)
            parsed.append(
                {
                    "id": f"{split}/{name}",
                    "path": str(path),
                    "class": label,
                    "split": split,
                }
            )
    classes = CLASSES.get(task) or sorted({r["class"] for r in parsed})
    if not all(any(r["split"] == s for r in parsed) for s in ("train", "test")):
        raise ValueError("Both splits must be nonempty")
    return [{**r, "label": classes.index(r["class"])} for r in parsed], classes


def _random_partition(rows, clients, variation, rng):
    if len(rows) < clients:
        raise ValueError(
            "Too few examples for every client; increase label budget or reduce clients"
        )
    # Integer allocation near N/K; exact total and no dropped tail.
    mean = len(rows) / clients
    lower = max(1, math.floor(mean * (1 - variation)))
    upper = max(lower + 1, math.ceil(mean * (1 + variation)))
    counts = np.full(clients, lower, dtype=int)
    while counts.sum() < len(rows):
        choices = np.flatnonzero(counts < upper)
        counts[rng.choice(choices)] += 1
    order = rng.permutation(len(rows))
    ends = np.r_[0, np.cumsum(counts)]
    return [
        [rows[i].copy() for i in order[ends[k] : ends[k + 1]]] for k in range(clients)
    ]


def partition(
    rows,
    clients,
    labeled_fraction,
    unlabeled_fraction,
    variation,
    class_skew,
    seed,
    supervised_only=False,
):
    rng = np.random.default_rng(seed)
    rng_u = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    labeled = []
    unlabeled = []
    classes = sorted({r["label"] for r in rows})
    for label in classes:
        group = [r for r in rows if r["label"] == label]
        order = rng.permutation(len(group))
        count = int(len(group) * labeled_fraction)
        labeled.extend(group[i].copy() for i in order[:count])
        unlabeled.extend(
            {k: v for k, v in group[i].items() if k != "class"} | {"label": -1}
            for i in order[count:]
        )
    if class_skew:
        if any(not any(r["label"] == c for r in labeled) for c in classes):
            raise ValueError(
                "Class-skew mode requires labeled examples for every class"
            )
        assignments = np.array_split(
            rng.permutation(max(clients, len(classes))), min(clients, len(classes))
        )
        out = [[] for _ in range(clients)]
        if clients <= len(classes):
            for cid, labels in enumerate(assignments):
                out[cid] = [
                    r for r in labeled if r["label"] in [classes[x] for x in labels]
                ]
        else:
            # Each client owns one class; a class may be split among several clients.
            for class_index, ids in enumerate(assignments):
                group = [r for r in labeled if r["label"] == classes[class_index]]
                if len(group) < len(ids):
                    raise ValueError(
                        "Insufficient labeled data for class-skew client count"
                    )
                for cid, indices in zip(
                    ids, np.array_split(rng.permutation(len(group)), len(ids))
                ):
                    out[cid] = [group[i] for i in indices]
    else:
        out = _random_partition(labeled, clients, variation, rng)
    if not supervised_only and unlabeled_fraction > 0:
        count = int(len(unlabeled) * unlabeled_fraction)
        if count:
            subset = [unlabeled[i] for i in rng_u.permutation(len(unlabeled))[:count]]
            uclients = _random_partition(subset, clients, variation, rng_u)
            out = [l + u for l, u in zip(out, uclients)]
    if any(not any(r["label"] >= 0 for r in c) for c in out):
        raise ValueError("Every client requires labeled data")
    return out


def read_wave(path):
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // divisor, rate // divisor)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f"Invalid audio: {path}")
    return wave.astype(np.float32)


def features(wave, training, rng):
    wave = np.asarray(wave, dtype=np.float32)
    if len(wave) < 16000:
        extra = 16000 - len(wave)
        wave = np.pad(wave, (extra // 2, extra - extra // 2))
    if training:
        wave = wave / max(np.linalg.norm(wave), 1e-9)
        start = int(rng.integers(len(wave) - 16000 + 1))
        segments = wave[start : start + 16000][None]
    else:
        segments = wave[: len(wave) // 16000 * 16000].reshape(-1, 16000)
        segments = segments / np.maximum(
            np.linalg.norm(segments, axis=1, keepdims=True), 1e-9
        )
    spectrum = tf.abs(tf.signal.stft(segments, 400, 160, fft_length=1024))
    mel = tf.signal.linear_to_mel_weight_matrix(64, 513, 16000, 60.0, 7800.0)
    x = tf.math.log(tf.clip_by_value(tf.tensordot(spectrum, mel, 1), 1e-5, 1e8))[
        ..., None
    ].numpy()
    return x[0] if training else x
