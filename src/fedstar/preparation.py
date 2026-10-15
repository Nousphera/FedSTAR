"""One-call preparation of real FedSTAR audio corpora and their fixed splits."""
import argparse
import csv
from http.client import HTTPException
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from .data import (CLASSES, DATASETS, WORDS, check_records, checksum,
                   manifest_fingerprint, verify_dataset)

SPEECH_BASE = "https://storage.googleapis.com/download.tensorflow.org/data/"
SPEECH_ARCHIVES = {
    "train": ("speech_commands_v0.02.tar.gz", "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"),
    "test": ("speech_commands_test_set_v0.02.tar.gz", "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df"),
}
AMBIENT_PAGE = "https://www.esense.io/datasets/ambientacousticcontext/index.html"
AMBIENT_URL = "https://drive.usercontent.google.com/download?id=1QzplAZKooora5TRTa1Ai0AaBjsqS_bpH&export=download&confirm=t"
VOXFORGE_BASE = "http://www.repository.voxforge1.org/downloads/"
FSD_BASE = "https://zenodo.org/records/4060432/files/"
FSD_FILES = {
    "FSD50K.dev_audio.z01": "faa7cf4cc076fc34a44a479a5ed862a3",
    "FSD50K.dev_audio.z02": "8f9b66153e68571164fb1315d00bc7bc",
    "FSD50K.dev_audio.z03": "1196ef47d267a993d30fa98af54b7159",
    "FSD50K.dev_audio.z04": "d088ac4e11ba53daf9f7574c11cccac9",
    "FSD50K.dev_audio.z05": "81356521aa159accd3c35de22da28c7f",
    "FSD50K.dev_audio.zip": "c480d119b8f7a7e32fdb58f3ea4d6c5a",
    "FSD50K.ground_truth.zip": "ca27382c195e37d2269c4c866dd73485",
    "FSD50K.metadata.zip": "b9ea0c829a411c1d42adb9da539ed237",
    "FSD50K.doc.zip": "3516162b82dc2945d3e7feba0904e800",
}


def write_json(path, value):
    """Publish a complete JSON file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download(url, destination, expected=None, algorithm="sha256"):
    """Resume downloads, verify upstream digests when available, and detect stale caches."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + ".download.json")
    if destination.exists():
        saved = json.loads(receipt.read_text()) if receipt.is_file() else {}
        target = expected or saved.get("sha256")
        if destination.is_symlink() or not target or checksum(destination, algorithm if expected else "sha256") != target:
            raise ValueError(f"Checksum mismatch or unverified download: {destination}")
        return destination
    partial = destination.with_name(destination.name + ".part")
    partial_receipt = partial.with_name(partial.name + ".json")
    if partial.is_symlink():
        raise ValueError(f"Download path is a symbolic link: {partial}")
    if expected and partial.is_file() and checksum(partial, algorithm) == expected:
        partial.replace(destination)
        write_json(receipt, {"url": url, "sha256": checksum(destination), "upstream_checksum": expected})
        return destination
    offset = partial.stat().st_size if partial.exists() else 0
    saved = json.loads(partial_receipt.read_text()) if partial_receipt.is_file() else {}
    # An upstream digest protects a resumed object; otherwise require a server validator.
    if not expected and (saved.get("url") != url or not saved.get("validator")):
        offset = 0
    headers = {"User-Agent": "FedSTAR-data/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if saved.get("validator"):
            headers["If-Range"] = saved["validator"]
    print(f"Downloading {destination.name}", flush=True)
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError(f"Provider returned an access/confirmation page instead of audio archive: {url}")
        resume = offset > 0 and response.status == 206
        if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Server returned an incorrect download range")
        length = response.headers.get("Content-Length")
        if length and int(length) + 64 * 1024**2 > shutil.disk_usage(destination.parent).free:
            raise OSError(f"Insufficient free disk space for {destination.name}; choose --data-root on a larger disk")
        write_json(partial_receipt, {"url": url, "validator": response.headers.get("ETag") or response.headers.get("Last-Modified")})
        with partial.open("ab" if resume else "wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if length and partial.stat().st_size != int(length) + (offset if resume else 0):
            raise OSError(f"Incomplete download retained for resume: {partial}")
    if expected and checksum(partial, algorithm) != expected:
        raise ValueError(f"Checksum mismatch: {partial}; remove this damaged partial file and retry")
    partial.replace(destination)
    write_json(receipt, {"url": url, "sha256": checksum(destination), "upstream_checksum": expected})
    partial_receipt.unlink(missing_ok=True)
    return destination


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"Unsafe relative path: {name}")
    return Path(*path.parts)


def extract_archive(archive, destination):
    """Extract atomically; never follow archive links or overwrite untracked data."""
    archive, destination = Path(archive), Path(destination)
    digest = checksum(archive)
    marker = destination / ".archive.json"
    if marker.is_file():
        if json.loads(marker.read_text()).get("sha256") != digest:
            raise ValueError(f"Extraction belongs to another archive: {destination}")
        return destination
    if destination.exists():
        raise FileExistsError(f"Unverified extraction already exists: {destination}; use --source-dir to import it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        seen = set()

        def target_for(name):
            if name in (".", "./"):
                return stage
            relative = safe_member(name)
            if relative in seen:
                raise ValueError(f"Duplicate archive member: {name}")
            seen.add(relative)
            return stage / relative

        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                for item in source.infolist():
                    target = target_for(item.filename)
                    mode = (item.external_attr >> 16) & 0o170000
                    if mode not in (0, 0o040000, 0o100000):
                        raise ValueError(f"Unsafe archive link/special file: {item.filename}")
                    if item.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with source.open(item) as reader, target.open("xb") as writer:
                            shutil.copyfileobj(reader, writer)
        else:
            with tarfile.open(archive) as source:
                for item in source:
                    target = target_for(item.name)
                    if item.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif item.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with source.extractfile(item) as reader, target.open("xb") as writer:
                            shutil.copyfileobj(reader, writer)
                    else:
                        raise ValueError(f"Unsafe archive link/special file: {item.name}")
        write_json(stage / ".archive.json", {"sha256": digest})
        stage.rename(destination)
    return destination


def finish_manifest(dataset, directory, records, metadata, metadata_files=()):
    """Write only after all real audio and split integrity checks have succeeded."""
    directory = Path(directory).resolve()
    if (directory / "manifest.json").exists():
        raise FileExistsError(f"Prepared manifest already exists: {directory}; choose a new data root")
    rows = check_records(dataset, records, populate=True)
    value = {"schema": 2, "dataset": dataset, "classes": CLASSES[dataset],
             "records": rows, "preparation": metadata,
             "metadata_files": [{"path": str(Path(p).resolve()), "sha256": checksum(p)} for p in metadata_files]}
    value["fingerprint"] = manifest_fingerprint(value)
    write_json(directory / "manifest.json", value)
    return verify_dataset(dataset, directory)


def _default_splits(dataset):
    # Source checkout and wheel installations are both supported.
    candidates = [Path(__file__).resolve().parents[2] / "data_splits" / dataset,
                  Path(__file__).resolve().parent / "data_splits" / dataset]
    for path in candidates:
        if (path / "train_split.txt").is_file():
            return path
    raise FileNotFoundError("Dataset split files are missing; pass --split-dir pointing to data_splits/<dataset>")


def read_splits(dataset, roots, split_dir):
    """Keep supplied row order and shared class IDs; never omit missing rows."""
    rows = []
    missing = []
    for split in ("train", "test"):
        root = Path(roots[split]).resolve()
        for line in (Path(split_dir) / f"{split}_split.txt").read_text().splitlines():
            if not line.strip():
                continue
            if dataset == "ambient_context":
                name, label = line.split("\t")
                label = label.strip()
            else:
                name = line.strip()
                label = PurePosixPath(name).parts[0]
                if dataset == "speech_commands":
                    label = "silence" if label in {"_silence_", "silence", "training_silence"} else label
                    label = label if label in CLASSES[dataset] else "unknown"
            relative = safe_member(name)
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Split path escapes source root: {name}")
            if label not in CLASSES[dataset]:
                raise ValueError(f"Unknown {dataset} class label: {label}")
            if not path.is_file():
                missing.append(str(path))
            rows.append({"id": f"{split}/{relative.as_posix()}", "path": str(path),
                         "label": CLASSES[dataset].index(label), "split": split})
    if missing:
        detail = "\n".join(missing[:5])
        note = ("\nThe historical Speech Commands split needs its 668 original _silence_ clips. "
                "Use --protocol official for the recorded background-noise recipe." if dataset == "speech_commands" else "")
        raise FileNotFoundError(f"{dataset}: {len(missing)} split recordings are missing:\n{detail}{note}")
    return rows


def _space(root, gib, existing=()):
    root.mkdir(parents=True, exist_ok=True)
    staged, seen = 0, set()
    candidates = [candidate for item in existing
                  for candidate in (Path(item), Path(item).with_name(Path(item).name + ".part"))]
    for item in candidates:
        for path in ([item] if item.is_file() else item.rglob("*") if item.is_dir() else []):
            if path.is_file() and not path.is_symlink():
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    staged += stat.st_size
                    seen.add(identity)
    remaining = max(0, gib * 1024**3 - staged)
    if shutil.disk_usage(root).free < remaining:
        raise OSError(f"Preparation needs approximately {remaining / 1024**3:.1f} GiB more free at {root}; "
                      "use --data-root on a larger disk or --source-dir for an existing corpus")


def _speech_roots(source):
    for train, test in [(source / "Data/Train", source / "Data/Test"),
                        (source / "train", source / "test"),
                        (source / "speech_commands_v0.02", source / "speech_commands_test_set_v0.02")]:
        if train.is_dir() and test.is_dir():
            return {"train": train, "test": test}
    raise FileNotFoundError(f"Speech Commands source must contain train/ and test/ (or Data/Train, Data/Test): {source}. "
                            f"Official archives: {SPEECH_BASE}")


def _speech_official(roots, directory, seed):
    import numpy as np
    import soundfile as sf
    from .audio import read_wave
    train, test = roots["train"], roots["test"]
    excluded = set()
    metadata_files = [train / "validation_list.txt", train / "testing_list.txt"]
    for path in metadata_files:
        excluded.update(path.read_text().splitlines())
    rows = []
    for split, base in roots.items():
        for path in sorted(base.glob("*/*.wav")):
            name = path.relative_to(base).as_posix()
            if split == "train" and (name in excluded or path.parent.name == "_background_noise_"):
                continue
            word = path.parent.name
            label = "silence" if word == "_silence_" else word if word in WORDS else "unknown"
            rows.append({"id": f"{split}/{name}", "path": str(path.resolve()),
                         "label": CLASSES["speech_commands"].index(label), "split": split})
    speakers = {s: {Path(r["path"]).stem.split("_nohash_")[0] for r in rows
                    if r["split"] == s and "_nohash_" in r["path"]} for s in ("train", "test")}
    if speakers["train"] & speakers["test"]:
        raise ValueError("Speech Commands training and test speakers overlap")
    noises = sorted((train / "_background_noise_").glob("*.wav"))
    if not noises:
        raise FileNotFoundError(f"Speech Commands background noise missing: {train / '_background_noise_'}")
    # Keep the released split's silence count, with an explicit newly specified recipe.
    rng = np.random.default_rng(seed)
    noise_arrays = [read_wave(p) for p in noises]
    silence_dir = directory / "silence"
    recipe = []
    if not silence_dir.is_dir():
        silence_dir.mkdir(parents=True)
    for index in range(668):
        source = int(rng.integers(len(noises)))
        wave = noise_arrays[source]
        if len(wave) < 16000:
            raise ValueError(f"Background noise is shorter than one second: {noises[source]}")
        start = int(rng.integers(len(wave) - 16000 + 1))
        path = silence_dir / f"{index:06d}.wav"
        if path.exists():
            stored, rate = sf.read(path, dtype="int16")
            expected = np.floor(np.clip(wave[start:start + 16000], -1, 1 - 1 / 32768) * 32768).astype("int16")
            if rate != 16000 or not np.array_equal(stored, expected):
                raise ValueError(f"Existing silence clip belongs to another recipe or changed: {path}")
        else:
            sf.write(path, wave[start:start + 16000], 16000, subtype="PCM_16")
        rows.append({"id": f"train/silence/{index:06d}.wav", "path": str(path), "label": 10, "split": "train"})
        recipe.append({"file": path.name, "source": str(noises[source]), "start_16khz": start})
    return rows, {"protocol": "official-heldout-seeded-noise-v1", "seed": seed,
                  "silence_count": 668, "silence_recipe": recipe,
                  "split": "Official validation/testing lists excluded; separate official test archive"}, metadata_files + noises


def _ambient_roots(source, split_dir):
    # Author archive nests each one-second WAV under its original AudioSet clip directory.
    wanted = {line.split("\t")[0] for split in ("train", "test")
              for line in (split_dir / f"{split}_split.txt").read_text().splitlines() if line.strip()}
    found = {}
    for path in source.rglob("*.wav"):
        if path.name in wanted:
            if path.name in found:
                raise ValueError(f"Ambiguous Ambient recording name in source: {path.name}")
            found[path.name] = path
    missing = sorted(wanted - found.keys())
    if missing:
        raise FileNotFoundError(f"ambient_context: {len(missing)} recordings missing from {source}, e.g. {missing[:3]}. "
                                f"Download the author archive from {AMBIENT_PAGE}; preserve its nested WAV folders.")
    rows = []
    for split in ("train", "test"):
        for line in (split_dir / f"{split}_split.txt").read_text().splitlines():
            if not line.strip():
                continue
            name, label = line.split("\t")
            safe_member(name)
            label = label.strip()
            if label not in CLASSES["ambient_context"]:
                raise ValueError(f"Unknown Ambient class label: {label}")
            path = found[name].resolve()
            if not path.is_relative_to(source.resolve()):
                raise ValueError(f"Ambient path escapes source root: {path}")
            rows.append({"id": f"{split}/{name}", "path": str(path),
                         "label": CLASSES["ambient_context"].index(label), "split": split})
    return rows


def _download_voxforge(root, split_dir):
    groups = set()
    for split in ("train", "test"):
        for line in (split_dir / f"{split}_split.txt").read_text().splitlines():
            path = safe_member(line.strip())
            if path.parts[0] not in CLASSES["voxforge"] or len(path.parts) < 8:
                raise ValueError(f"Unexpected VoxForge corpus path: {line}")
            groups.add(Path(*path.parts[:6]))
    source = root / "raw/voxforge"
    if any(not (source / group).is_dir() for group in groups):
        _space(root, 50, [root / "downloads/voxforge", source])
    archives = []
    for group in sorted(groups):
        prefix, name = group.parent, group.name
        remote = prefix.as_posix().replace("en/Trunk/", "SpeechCorpus/Trunk/") + f"/{name}.tgz"
        archive = download(VOXFORGE_BASE + remote, root / "downloads/voxforge" / prefix / f"{name}.tgz")
        extracted = extract_archive(archive, source / ".archives" / group)
        original = extracted / name
        if not original.is_dir():
            raise FileNotFoundError(f"VoxForge archive lacks {name}/: {archive}")
        target = source / group
        target.parent.mkdir(parents=True, exist_ok=True)
        # The link stays inside the source root and avoids a second corpus-sized copy.
        if not target.exists():
            target.symlink_to(os.path.relpath(original, target.parent), target_is_directory=True)
        archives.append(archive)
    return source, archives


def _prepare_fsd(root, source):
    if source is None:
        _space(root, 65, [root / "downloads/fsd50k", root / "raw/fsd50k", root / "extracted/fsd50k"])
        if not shutil.which("zip"):
            raise OSError("FSD50K multipart archives need the 'zip' command. Install zip or download/extract "
                          "https://zenodo.org/records/4060432 and pass --source-dir with FSD50K.dev_audio/ and FSD50K.ground_truth/dev.csv")
        downloads = root / "downloads/fsd50k"
        for name, digest in FSD_FILES.items():
            download(FSD_BASE + name, downloads / name, digest, "md5")
        merged = downloads / "dev-unsplit.zip"
        if not merged.exists():
            temporary = downloads / "dev-unsplit.partial.zip"
            temporary.unlink(missing_ok=True)
            subprocess.run(["zip", "-s", "0", str(downloads / "FSD50K.dev_audio.zip"), "--out", str(temporary)], check=True)
            temporary.replace(merged)
        source = extract_archive(merged, root / "raw/fsd50k")
        for name in ("FSD50K.ground_truth.zip", "FSD50K.metadata.zip", "FSD50K.doc.zip"):
            extracted = extract_archive(downloads / name, root / "extracted/fsd50k" / name)
            folder = name.removesuffix(".zip")
            if not (source / folder).is_dir():
                shutil.copytree(extracted / folder, source / folder)
    source = Path(source).resolve()
    metadata = source / "FSD50K.ground_truth/dev.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"FSD50K requires {metadata} and FSD50K.dev_audio/*.wav. Download/extract "
                                "https://zenodo.org/records/4060432; evaluation audio is excluded.")
    rows = []
    with metadata.open(newline="") as stream:
        for record in csv.DictReader(stream):
            name = record["fname"]
            if not name.isdigit() or record.get("split") not in {"train", "val"}:
                raise ValueError(f"Invalid FSD50K development metadata row: {record}")
            path = source / "FSD50K.dev_audio" / f"{name}.wav"
            if not path.resolve().is_relative_to((source / "FSD50K.dev_audio").resolve()):
                raise ValueError(f"FSD50K path escapes development audio: {path}")
            rows.append({"id": f"dev/{name}", "path": str(path), "label": -1, "split": "train"})
    return rows, {"protocol": "fsd50k-development-only", "source": "https://zenodo.org/records/4060432",
                  "split": "All development train/val clips; no evaluation clips; no class targets"}, [metadata]


def prepare_dataset(dataset, root="data", *, source_dir=None, split_dir=None, protocol="official", seed=2021, validate=False):
    """Prepare once, then validate and reuse; supplied audio is never modified."""
    if dataset not in CLASSES:
        raise ValueError(f"Unknown dataset: {dataset}")
    if protocol not in {"official", "supplied"} or type(seed) is not int or seed < 0:
        raise ValueError("protocol must be official/supplied and seed a nonnegative integer")
    root = Path(root).expanduser().resolve()
    directory = root / dataset
    if (directory / "manifest.json").is_file() and not validate and dataset == "speech_commands":
        previous = json.loads((directory / "manifest.json").read_text()).get("preparation", {})
        expected = "supplied-split-lists" if protocol == "supplied" or split_dir is not None else "official-heldout-seeded-noise-v1"
        if previous.get("protocol") != expected or (expected == "official-heldout-seeded-noise-v1" and previous.get("seed") != seed):
            raise ValueError("Speech Commands is already prepared with a different protocol/seed; choose a new --data-root")
    if (directory / "manifest.json").is_file() or validate:
        return verify_dataset(dataset, root)
    source = Path(source_dir).expanduser().resolve() if source_dir else None
    if source is not None and not source.is_dir():
        raise FileNotFoundError(f"{dataset} source directory does not exist: {source}")
    splits = Path(split_dir).expanduser().resolve() if split_dir else None
    sources, metadata_files = [], []
    if dataset == "speech_commands":
        if source is None:
            source = root / "raw/speech_commands"
            if not all((source / s).is_dir() for s in ("train", "test")):
                _space(root, 7, [source] + [root / "downloads" / entry[0] for entry in SPEECH_ARCHIVES.values()])
            for split, (name, digest) in SPEECH_ARCHIVES.items():
                archive = download(SPEECH_BASE + name, root / "downloads" / name, digest)
                extract_archive(archive, source / split)
                sources.append({"url": SPEECH_BASE + name, "sha256": digest})
        roots = _speech_roots(source)
        if protocol == "supplied" or splits is not None:
            splits = splits or _default_splits(dataset)
            rows = read_splits(dataset, roots, splits)
            metadata = {"protocol": "supplied-split-lists", "source": str(source)}
            metadata_files = [splits / f"{s}_split.txt" for s in ("train", "test")]
        else:
            rows, metadata, metadata_files = _speech_official(roots, directory, seed)
    elif dataset in ("ambient_context", "voxforge"):
        splits = splits or _default_splits(dataset)
        metadata_files = [splits / f"{s}_split.txt" for s in ("train", "test")]
        if dataset == "ambient_context":
            if source is None:
                _space(root, 7, [root / "downloads/audioset_1sec_v1.tar.gz", root / "raw/ambient_context"])
                archive = download(AMBIENT_URL, root / "downloads/audioset_1sec_v1.tar.gz")
                source = extract_archive(archive, root / "raw/ambient_context")
                sources.append({"url": AMBIENT_URL, "sha256": checksum(archive)})
            rows = _ambient_roots(source, splits)
        else:
            if source is None:
                source, archives = _download_voxforge(root, splits)
                sources = [{"archive": str(p), "sha256": checksum(p)} for p in archives]
            rows = read_splits(dataset, {"train": source, "test": source}, splits)
        metadata = {"protocol": "supplied-split-lists", "source": str(source)}
        if dataset == "ambient_context":
            metadata["classes_note"] = "Released splits have nine semantic non-speech classes after trimming whitespace; paper Table 1 states ten."
    else:
        rows, metadata, metadata_files = _prepare_fsd(root, source)
    metadata["downloads"] = sources
    return finish_manifest(dataset, directory, rows, metadata, metadata_files)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and prepare real FedSTAR datasets; omitted --dataset prepares all downstream datasets.")
    parser.add_argument("--dataset", choices=CLASSES, help="fsd50k is optional pretraining data")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--source-dir", type=Path, help="Import an existing corpus; requires --dataset")
    parser.add_argument("--split-dir", type=Path, help="Directory containing train_split.txt and test_split.txt; requires --dataset")
    parser.add_argument("--protocol", choices=("official", "supplied"), default="official", help="Speech Commands: official heldouts plus seeded silence, or exact supplied split files")
    parser.add_argument("--seed", type=int, default=2021, help="Seed for the official Speech Commands silence recipe")
    parser.add_argument("--validate", action="store_true", help="Verify prepared data without downloading")
    args = parser.parse_args(argv)
    if not args.dataset and (args.source_dir or args.split_dir):
        parser.error("--source-dir and --split-dir require --dataset")
    failures = []
    for dataset in (args.dataset,) if args.dataset else DATASETS:
        try:
            bundle = prepare_dataset(dataset, args.data_root, source_dir=args.source_dir, split_dir=args.split_dir,
                                     protocol=args.protocol, seed=args.seed, validate=args.validate)
            print(f"{dataset}: ready ({len(bundle.train)} train, {len(bundle.test)} test; {len(bundle.classes)} classes)")
        except (OSError, ValueError, HTTPError, HTTPException, URLError, tarfile.TarError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
            failures.append(dataset)
            print(f"{dataset}: {error}")
            if dataset == "ambient_context":
                print(f"Provider: {AMBIENT_PAGE}\nDownload: https://drive.google.com/file/d/1QzplAZKooora5TRTa1Ai0AaBjsqS_bpH/view\n"
                      "Extract the archive and rerun with --source-dir PATH (nested <clip>/<clip>_<index>.wav supported).")
            elif dataset == "voxforge":
                print(f"Provider: https://www.voxforge.org/home/Downloads\nDownloads: {VOXFORGE_BASE}\n"
                      "Expected --source-dir layout: <language>/Trunk/Audio/Main/16kHz_16bit/<submission>/wav/<file>.wav")
            elif dataset == "fsd50k":
                print("Provider: https://zenodo.org/records/4060432\nExpected --source-dir layout: FSD50K.dev_audio/*.wav and FSD50K.ground_truth/dev.csv")
    return 1 if failures else 0
