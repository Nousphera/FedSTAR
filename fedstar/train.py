"""Shared trainer for sequential FedAvg and Flower ClientApps."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import time
import numpy as np
import tensorflow as tf
from .data import features, partition, read_wave
from .method import aggregate, confidence_threshold, self_training_loss
from .model import build_model


@dataclass(frozen=True)
class RunConfig:
    dataset: str = "synthetic"
    manifest: str = "data/manifest.json"
    clients: int = 2
    labeled_fraction: float = 0.1
    unlabeled_fraction: float = 1.0
    quantity_variation: float = 0.25
    class_skew: bool = False
    supervised_only: bool = False
    rounds: int = 100
    participation: float = 0.8
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.001
    beta: float = 0.5
    mode: str = "paper"
    seed: int = 2021
    synthetic_per_class: int = 20
    pretrained_encoder: str | None = None
    output: str = "runs/experiment"

    def __post_init__(self):
        if self.dataset not in {
            "synthetic",
            "speech_commands",
            "voxforge",
            "ambient_context",
        }:
            raise ValueError("Unknown dataset")
        if self.mode not in {"paper", "source-loss"}:
            raise ValueError("Unknown loss mode")
        if any(
            not isinstance(x, int) or x < 1
            for x in (
                self.clients,
                self.rounds,
                self.local_epochs,
                self.batch_size,
                self.synthetic_per_class,
            )
        ):
            raise ValueError("Counts must be positive integers")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch_size must be even and >=2")
        if not (
            0 < self.labeled_fraction <= 1
            and 0 <= self.unlabeled_fraction <= 1
            and 0 < self.participation <= 1
            and 0 <= self.quantity_variation < 1
            and 0 <= self.beta <= 1
            and self.learning_rate > 0
            and self.seed >= 0
        ):
            raise ValueError("Invalid training hyperparameters")

    @property
    def temperature(self):
        return 4.0 if self.mode == "paper" else 1.0


def load_config(path, **overrides):
    config_path = Path(path).resolve()
    value = json.loads(config_path.read_text())
    # Shipped configs live under configs/; keep their data paths tied to the
    # checkout even when Flower executes a bundled app from a different cwd.
    base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    for key in ("manifest", "pretrained_encoder", "output"):
        if value.get(key) and not Path(value[key]).is_absolute():
            value[key] = str((base / value[key]).resolve())
    value.update({k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**value)


def make_data(config):
    if config.dataset == "synthetic":
        rng = np.random.default_rng(config.seed)
        rows = []
        test = []
        classes = ["low", "mid", "high"]
        for split, total in [
            ("train", config.synthetic_per_class * len(classes)),
            ("test", 6),
        ]:
            for i in range(total):
                label = i % len(classes)
                samples = 16000 * (2 if split == "test" else 1)
                t = np.arange(samples) / 16000
                wave = (
                    0.2 * np.sin(2 * np.pi * (200 + label * 180) * t)
                    + rng.normal(0, 0.02, samples)
                ).astype(np.float32)
                (rows if split == "train" else test).append(
                    {"id": f"{split}-{i}", "label": label, "_wave": wave}
                )
    else:
        value = json.loads(Path(config.manifest).read_text())
        if value["task"] != config.dataset:
            raise ValueError("Manifest task differs from config.dataset")
        from .preparation import check_records
        classes = value["classes"]
        check_records(value["records"], classes, ("train", "test"))
        from .data import CLASSES
        if config.dataset in CLASSES and classes != CLASSES[config.dataset]:
            raise ValueError("Manifest class order differs from task labels")
        rows = [r for r in value["records"] if r["split"] == "train"]
        test = [r for r in value["records"] if r["split"] == "test"]
    clients = partition(
        rows,
        config.clients,
        config.labeled_fraction,
        config.unlabeled_fraction,
        config.quantity_variation,
        config.class_skew,
        config.seed,
        config.supervised_only,
    )
    return clients, test, classes


def waveform(record):
    return record["_wave"] if "_wave" in record else read_wave(record["path"])


def initial_model(config, num_classes):
    model = build_model(num_classes)
    if config.pretrained_encoder:
        encoder = tf.keras.models.load_model(config.pretrained_encoder)
        model.get_layer("encoder").set_weights(encoder.get_weights())
    return model


def local_train(weights, records, num_classes, config, completed, client_id):
    tf.keras.backend.clear_session()
    seed = int(
        np.random.SeedSequence([config.seed, completed, client_id]).generate_state(1)[0]
    )
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    rng = np.random.default_rng(seed)
    model = build_model(num_classes)
    model.set_weights(weights)
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    threshold = confidence_threshold(completed, config.rounds)
    labeled = [r for r in records if r["label"] >= 0]
    unlabeled = [r for r in records if r["label"] == -1]
    if not labeled:
        raise ValueError("Every participating client needs labeled examples")
    loss_sum = 0.0
    accepted = num_unlabeled = steps = 0
    for _ in range(config.local_epochs):
        order = rng.permutation(len(unlabeled) if unlabeled else len(labeled))
        lorder = rng.permutation(len(labeled))
        size = config.batch_size // 2 if unlabeled else config.batch_size
        for offset in range(0, len(order), size):
            indices = order[offset : offset + size]
            batch = (
                (
                    [
                        labeled[lorder[i % len(labeled)]]
                        for i in range(offset, offset + len(indices))
                    ]
                    + [unlabeled[i] for i in indices]
                )
                if unlabeled
                else [labeled[i] for i in indices]
            )
            x = np.stack([features(waveform(r), True, rng) for r in batch])
            y = np.array([r["label"] for r in batch], np.int32)
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss, info = self_training_loss(
                    logits,
                    y,
                    threshold,
                    config.temperature,
                    config.beta,
                    config.mode if unlabeled else "paper",
                )
                loss += tf.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            tf.debugging.assert_all_finite(loss, "Nonfinite local objective")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_sum += float(loss)
            accepted += int(info["accepted"])
            num_unlabeled += int(info["unlabeled"])
            steps += 1
    return (
        model.get_weights(),
        len(records),
        {
            "loss": loss_sum / steps,
            "threshold": threshold,
            "accepted": accepted,
            "unlabeled": num_unlabeled,
            "steps": steps,
        },
    )


def evaluate(model, records):
    labels = []
    predictions = []
    losses = []
    for row in records:
        logits = model(
            features(waveform(row), False, np.random.default_rng(0)), training=False
        )
        probability = tf.reduce_mean(tf.nn.softmax(logits), axis=0).numpy()
        labels.append(row["label"])
        predictions.append(int(probability.argmax()))
        losses.append(-np.log(max(float(probability[row["label"]]), 1e-12)))
    return {
        "accuracy": float(np.mean(np.array(labels) == predictions)),
        "eval_loss": float(np.mean(losses)),
    }


def prepare_output(config, clients, test, classes, engine):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(asdict(config) | {"temperature": config.temperature}, indent=2)
        + "\n"
    )
    public_rows = [
        [{k: v for k, v in r.items() if k != "_wave"} for r in c] for c in clients
    ]
    (output / "partitions.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "clients": public_rows,
                "test_ids": [r["id"] for r in test],
            },
            indent=2,
        )
        + "\n"
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "engine": engine,
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "devices": [d.name for d in tf.config.list_physical_devices()],
    }
    if config.dataset != "synthetic":
        environment["manifest_sha256"] = hashlib.sha256(
            Path(config.manifest).read_bytes()
        ).hexdigest()
    if config.pretrained_encoder:
        environment["encoder_sha256"] = hashlib.sha256(
            Path(config.pretrained_encoder).read_bytes()
        ).hexdigest()
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    return output


def run(config):
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()
    clients, test, classes = make_data(config)
    output = prepare_output(config, clients, test, classes, "sequential-fedavg")
    model = initial_model(config, len(classes))
    weights = model.get_weights()
    rng = np.random.default_rng(config.seed)
    history = []
    start = time.monotonic()
    for round_index in range(config.rounds):
        selected = sorted(
            rng.choice(
                config.clients,
                max(1, int(config.clients * config.participation)),
                replace=False,
            ).tolist()
        )
        updates = []
        training = []
        for cid in selected:
            update, count, metrics = local_train(
                weights, clients[cid], len(classes), config, round_index, cid
            )
            updates.append((update, count))
            training.append({"client": cid, "num_examples": count, **metrics})
        weights = aggregate(updates)
        model.set_weights(weights)
        result = {
            "round": round_index + 1,
            "selected_clients": selected,
            "client_metrics": training,
            **evaluate(model, test),
        }
        history.append(result)
        print(json.dumps(result), flush=True)
    model.save(output / "model.keras")
    result = {
        "dataset": config.dataset,
        "mode": config.mode,
        "seconds": time.monotonic() - start,
        "rounds": history,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
