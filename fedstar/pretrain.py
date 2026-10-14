"""Paper-derived FSD50K contrastive pretraining, not recovered historical code."""

from pathlib import Path
import json
import hashlib
import importlib.metadata
import platform
import numpy as np
import tensorflow as tf
from .data import read_wave, features
from .model import build_encoder, contrastive_loss


def pretrain(
    paths, output, epochs=500, batch_size=1024, seed=2021, learning_rate=0.001
):
    if len(paths) < 2 or batch_size < 2 or epochs < 1:
        raise ValueError("Need >=2 clips/batch and positive epochs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    rng = np.random.default_rng(seed)
    encoder = build_encoder()
    projector = tf.keras.layers.Dense(256)
    projector(tf.zeros((1, 128)))
    bilinear = tf.Variable(tf.eye(256), name="bilinear")
    variables = encoder.trainable_variables + projector.trainable_variables + [bilinear]
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(paths))
        losses = []
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            if len(ids) < 2:
                continue
            waves = [read_wave(paths[i]) for i in ids]
            first = np.stack([features(w, True, rng) for w in waves])
            second = np.stack([features(w, True, rng) for w in waves])
            with tf.GradientTape() as tape:
                a = projector(encoder(first, training=True))
                b = projector(encoder(second, training=True))
                loss = contrastive_loss(a, b, bilinear) + tf.add_n(encoder.losses)
            grads = tape.gradient(loss, variables)
            tf.debugging.assert_all_finite(loss, "Nonfinite contrastive objective")
            optimizer.apply_gradients(zip(grads, variables))
            losses.append(float(loss))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    encoder.save(output / "encoder.keras")
    (output / "clips.json").write_text(
        json.dumps(
            [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                }
                for path in paths
            ],
            indent=2,
        )
        + "\n"
    )
    (output / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    d.metadata["Name"]: d.version
                    for d in importlib.metadata.distributions()
                },
            },
            indent=2,
        )
        + "\n"
    )
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "paper_derived_reconstruction": True,
                "benchmark_reproduction": False,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "clips": len(paths),
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    return encoder
