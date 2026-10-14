import io
import json
import tarfile
import numpy as np
import pytest
import soundfile as sf
from fedstar.datasets import extract_tar, prepare_speech
from fedstar.preparation import check_records, write_json
from fedstar.data import read_wave
from fedstar.train import RunConfig, run, make_data


def test_archive_rejects_traversal_links_and_duplicate_paths(tmp_path):
    for index, name in enumerate(("../bad", "/absolute", "link", "duplicate")):
        archive = tmp_path / f"{index}.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            member = tarfile.TarInfo(name)
            if name == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "../outside"
            stream.addfile(member)
            if name == "duplicate":
                stream.addfile(member)
        destination = tmp_path / f"out-{index}"
        with pytest.raises(ValueError):
            extract_tar(archive, destination)
        assert not destination.exists()


def speech_fixture(tmp_path):
    rng = np.random.default_rng(2)
    for split in ("train", "test"):
        for label in ("yes", "no"):
            for index in range(4 if split == "train" else 1):
                path = tmp_path / split / label / f"{split}-{index}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, rng.normal(0, .1, 16000), 16000)
    for name in ("validation_list.txt", "testing_list.txt"):
        (tmp_path / "train" / name).write_text("")
    noise = tmp_path / "train/_background_noise_/noise.wav"
    noise.parent.mkdir()
    sf.write(noise, rng.normal(0, .1, 40000), 16000)


def test_speech_recipe_deterministic_and_feeds_actual_training(tmp_path):
    speech_fixture(tmp_path)
    a = prepare_speech(tmp_path / "train", tmp_path / "test", tmp_path / "a", silence_count=4)
    b = prepare_speech(tmp_path / "train", tmp_path / "test", tmp_path / "b", silence_count=4)
    va, vb = json.loads(a.read_text()), json.loads(b.read_text())
    assert [r["sha256"] for r in va["records"]] == [r["sha256"] for r in vb["records"]]
    assert va["preparation"]["historical_paper_split"] is False
    check_records(va["records"], va["classes"], ("train", "test"), decode=read_wave, verify_hashes=True)
    result = run(RunConfig(dataset="speech_commands", manifest=str(a), clients=2, rounds=1,
                          participation=1., labeled_fraction=.5, batch_size=4, output=str(tmp_path / "run")))
    assert np.isfinite(result["rounds"][0]["accuracy"])
    with pytest.raises(FileExistsError):
        prepare_speech(tmp_path / "train", tmp_path / "test", tmp_path / "a")


def test_atomic_write_and_training_reject_leaked_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    with pytest.raises(TypeError):
        write_json(path, object())
    assert not path.exists()
    audio = tmp_path / "x.wav"
    sf.write(audio, np.zeros(100), 16000)
    value = {"task": "ambient_context", "classes": ["a", "b"], "records": [
        {"id": split, "path": str(audio), "label": 0, "split": split} for split in ("train", "test")]}
    write_json(path, value)
    with pytest.raises(ValueError, match="Duplicate file"):
        make_data(RunConfig(dataset="ambient_context", manifest=str(path)))


def test_config_paths_stay_with_checkout_when_working_directory_changes(tmp_path, monkeypatch):
    from fedstar.train import load_config
    root = tmp_path / "checkout with spaces"
    (root / "configs").mkdir(parents=True)
    config = root / "configs/check.json"
    config.write_text(json.dumps({"manifest": "data/input.json", "output": "runs/check"}))
    monkeypatch.chdir(tmp_path)
    loaded = load_config(config)
    assert loaded.manifest == str(root / "data/input.json")
    assert loaded.output == str(root / "runs/check")


def test_manifest_rejects_same_inode_under_different_names(tmp_path):
    import os
    from fedstar.preparation import check_records
    first = tmp_path / "first.wav"
    first.write_bytes(b"fixture")
    second = tmp_path / "second.wav"
    os.link(first, second)
    rows = [{"id": "a", "path": str(first), "label": 0, "split": "train"},
            {"id": "b", "path": str(second), "label": 1, "split": "test"}]
    with pytest.raises(ValueError, match="Duplicate|leakage"):
        check_records(rows, ["a", "b"], ["train", "test"])
