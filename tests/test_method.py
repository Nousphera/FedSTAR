import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from fedstar.method import confidence_threshold, self_training_loss, aggregate
from fedstar.model import build_model, contrastive_loss
from fedstar.data import read_splits, partition, features, read_wave
from fedstar.train import RunConfig, make_data, run


def test_threshold_global_round_endpoints():
    assert confidence_threshold(0, 100) == pytest.approx(0.5)
    assert confidence_threshold(50, 100) == pytest.approx(0.7)
    assert confidence_threshold(100, 100) == pytest.approx(0.9)


def test_loss_temperature_mask_and_denominator():
    logits = tf.Variable([[3.0, 0.0], [4.0, 0.0], [0.0, 0.0]])
    with tf.GradientTape() as tape:
        loss, info = self_training_loss(logits, [0, -1, -1], 0.7, 4.0, 0.5)
    ce = np.log1p(np.exp(-4.0))
    assert float(info["unlabeled_loss"]) == pytest.approx(ce / 2, rel=1e-5)
    assert float(loss) == pytest.approx(np.log1p(np.exp(-3.0)) + 0.5 * ce / 2, rel=1e-5)
    assert int(info["accepted"]) == 1
    grad = tape.gradient(loss, logits).numpy()
    assert np.any(grad[1] != 0) and np.all(grad[2] == 0)
    _, cold = self_training_loss(logits, [0, -1, -1], 0.9, 1.0, 0.5)
    _, hot = self_training_loss(logits, [0, -1, -1], 0.9, 4.0, 0.5)
    assert int(cold["accepted"]) == 1 and int(hot["accepted"]) == 0


def test_zero_unlabeled_supervised_loss_and_exact_weighting():
    loss, _ = self_training_loss([[0.0, 0.0]], [1], 0.5, 4.0, 0.5)
    assert float(loss) == pytest.approx(np.log(2))
    result = aggregate(
        [([np.array([1.0], np.float32)], 3), ([np.array([5.0], np.float32)], 1)]
    )
    np.testing.assert_allclose(result[0], [2.0])
    with pytest.raises(ValueError):
        aggregate([([np.ones(1)], 0)])


def test_encoder_and_bilinear_gradients_save(tmp_path):
    model = build_model(3, dropout=0)
    x = tf.random.normal([2, 98, 64, 1])
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(model(x) ** 2)
    assert all(
        g is not None and np.isfinite(g).all()
        for g in tape.gradient(loss, model.trainable_variables)
    )
    assert [
        l.filters
        for l in model.get_layer("encoder").layers
        if isinstance(l, tf.keras.layers.Conv2D)
    ][::3] == [24, 32, 64, 128]
    path = tmp_path / "model.keras"
    model.save(path)
    np.testing.assert_allclose(model(x), tf.keras.models.load_model(path)(x))
    z = tf.Variable([[1.0, 0.0], [0.0, 1.0]])
    w = tf.Variable(tf.eye(2))
    with tf.GradientTape() as tape:
        loss = contrastive_loss(z, z, w)
    gs = tape.gradient(loss, [z, w])
    assert all(g is not None and np.any(g.numpy()) for g in gs)
    assert float(contrastive_loss(z, z, w)) < float(
        contrastive_loss(z, tf.reverse(z, [0]), w)
    )


def write_wav(path, rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = tf.sin(tf.range(rate, dtype=tf.float32) * 0.05)[:, None]
    tf.io.write_file(str(path), tf.audio.encode_wav(wav, rate))


@pytest.mark.parametrize(
    "task,train,test,expected",
    [
        (
            "speech_commands",
            "yes/a.wav\ncat/b.wav\n",
            "unknown/c.wav\nyes/d.wav\n",
            ["yes", "unknown"],
        ),
        ("voxforge", "de/a.wav\nen/b.wav\n", "en/c.wav\nde/d.wav\n", ["de", "en"]),
        (
            "ambient_context",
            "a.wav\tApplause\nb.wav\t Walk\n",
            "c.wav\tWalk\nd.wav\tApplause\n",
            ["Applause", "Walk"],
        ),
    ],
)
def test_all_split_adapters_stable_labels(tmp_path, task, train, test, expected):
    for split, text in [("train", train), ("test", test)]:
        (tmp_path / f"{split}.txt").write_text(text)
        for line in text.splitlines():
            write_wav(tmp_path / split / line.split("\t")[0])
    rows, classes = read_splits(
        task,
        tmp_path / "train",
        tmp_path / "test",
        tmp_path / "train.txt",
        tmp_path / "test.txt",
    )
    assert [classes[r["label"]] for r in rows if r["split"] == "train"] == expected
    assert [classes[r["label"]] for r in rows if r["split"] == "test"] == expected[::-1]
    with pytest.raises(FileNotFoundError):
        read_splits(
            task,
            tmp_path / "absent",
            tmp_path / "test",
            tmp_path / "train.txt",
            tmp_path / "test.txt",
        )


def test_missing_duplicate_and_overlap_fail(tmp_path):
    write_wav(tmp_path / "yes/a.wav")
    (tmp_path / "a.txt").write_text("yes/a.wav\nyes/a.wav\n")
    with pytest.raises(ValueError):
        read_splits(
            "speech_commands",
            tmp_path,
            tmp_path,
            tmp_path / "a.txt",
            tmp_path / "a.txt",
        )


def test_quantity_class_skew_and_no_label_leakage():
    rows = [{"id": str(i), "label": i % 3} for i in range(120)]
    a = partition(rows, 6, 0.5, 1.0, 0.25, True, 17)
    b = partition(rows, 6, 0.5, 1.0, 0.25, True, 17)
    assert a == b
    assert len({r["id"] for c in a for r in c}) == 120
    assert sum(len(c) for c in a) == 120
    assert all(len({r["label"] for r in c if r["label"] >= 0}) == 1 for c in a)
    plain = partition(rows, 6, 0.5, 1.0, 0.25, False, 17)
    assert [[r["id"] for r in c if r["label"] == -1] for c in a] == [
        [r["id"] for r in c if r["label"] == -1] for c in plain
    ]
    assert {int(r["id"]) % 3 for c in a for r in c if r["label"] == -1} == {0, 1, 2}
    assert sum(r["label"] >= 0 for c in a for r in c) == 60
    assert partition(rows, 6, 0.5, 0.5, 0.25, False, 17) != a


def test_audio_resampling_segmentation(tmp_path):
    write_wav(tmp_path / "48k.wav", 48000)
    wave = read_wave(tmp_path / "48k.wav")
    assert len(wave) == 16000
    rng = np.random.default_rng(7)
    assert features(wave, True, rng).shape == (98, 64, 1)
    assert features(np.tile(wave, 2), False, rng).shape == (2, 98, 64, 1)


def test_two_client_train_and_artifact_roundtrip(tmp_path):
    config = RunConfig(
        clients=2,
        rounds=1,
        participation=1,
        synthetic_per_class=4,
        labeled_fraction=0.5,
        batch_size=4,
        output=str(tmp_path / "run"),
    )
    result = run(config)
    assert result["dataset"] == config.dataset and result["mode"] == config.mode
    assert result["rounds"][0]["client_metrics"][0]["num_examples"] > 0
    assert np.isfinite(result["rounds"][0]["eval_loss"])
    assert (
        len(json.loads((tmp_path / "run/partitions.json").read_text())["clients"]) == 2
    )
    tf.keras.models.load_model(tmp_path / "run/model.keras")


def test_real_wav_training_and_encoder_transfer(tmp_path):
    from fedstar.pretrain import pretrain
    from fedstar.train import initial_model

    paths = []
    for split, n in [("train", 8), ("test", 2)]:
        lines = []
        for i in range(n):
            label = "de" if i % 2 else "en"
            name = f"{label}/{i}.wav"
            write_wav(tmp_path / split / name)
            paths.append(str(tmp_path / split / name))
            lines.append(name)
        (tmp_path / f"{split}.txt").write_text("\n".join(lines) + "\n")
    rows, classes = read_splits(
        "voxforge",
        tmp_path / "train",
        tmp_path / "test",
        tmp_path / "train.txt",
        tmp_path / "test.txt",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"task": "voxforge", "records": rows, "classes": classes})
    )
    encoder = pretrain(paths[:2], tmp_path / "pretrain", epochs=1, batch_size=2)
    metadata = json.loads((tmp_path / "pretrain/metrics.json").read_text())
    assert metadata["objective"] == "symmetric-bilinear-infonce"
    config = RunConfig(
        dataset="voxforge",
        manifest=str(manifest),
        clients=2,
        rounds=1,
        participation=1,
        labeled_fraction=0.5,
        batch_size=4,
        pretrained_encoder=str(tmp_path / "pretrain/encoder.keras"),
        output=str(tmp_path / "run"),
    )
    loaded = initial_model(config, 6).get_layer("encoder")
    for a, b in zip(encoder.get_weights(), loaded.get_weights()):
        np.testing.assert_array_equal(a, b)
    result = run(config)
    assert result["dataset"] == config.dataset and result["mode"] == config.mode
    assert all(np.isfinite(r["loss"]) for r in result["rounds"][0]["client_metrics"])
