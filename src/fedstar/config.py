"""Explicit parameters shared by preparation, clients and the server."""

import math
from pathlib import Path

DATASETS = ("speech_commands", "ambient_context", "voxforge")
METHODS = ("fedstar", "fedavg", "centralized")


def defaults():
    return {
        "method": "fedstar", "dataset": "speech_commands", "device": "cpu",
        "num-clients": 15, "num-rounds": 100, "participation": .8,
        "local-epochs": 1, "batch-size": 64, "learning-rate": .001,
        "labeled-fraction": .03, "unlabeled-fraction": 1., "quantity-variation": .25,
        "labeled-classes": 0, "class-variation": 0., "beta": .5, "temperature": 4.,
        "threshold-start": .5, "threshold-end": .9, "seed": 2021,
        "data-root": "data", "pretrained-encoder": "", "output": "results/fedstar.json",
        "torch-threads": 1, "num-workers": 0, "registration-timeout": 120., "reply-timeout": 86400.,
    }


def validate(cfg):
    cfg = dict(cfg)
    for key, value in cfg.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if cfg["dataset"] not in DATASETS or cfg["method"] not in METHODS:
        raise ValueError("Unknown dataset or method")
    for key in ("num-clients", "num-rounds", "local-epochs", "batch-size", "torch-threads"):
        if type(cfg[key]) != int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "num-workers", "labeled-classes"):
        if type(cfg[key]) != int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if (cfg["method"] == "fedstar" and cfg["labeled-fraction"] < 1 and cfg["unlabeled-fraction"] > 0
            and (cfg["batch-size"] < 2 or cfg["batch-size"] % 2)):
        raise ValueError("FedSTAR batch-size must be even: half labeled, half unlabeled")
    for key in ("labeled-fraction", "participation"):
        if not 0 < cfg[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    for key in ("unlabeled-fraction", "quantity-variation", "class-variation"):
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if not 0 <= cfg["threshold-start"] <= cfg["threshold-end"] <= 1:
        raise ValueError("Confidence thresholds must satisfy 0 <= start <= end <= 1")
    if cfg["beta"] < 0 or cfg["temperature"] <= 0 or cfg["learning-rate"] <= 0:
        raise ValueError("Invalid loss or optimizer parameters")
    if cfg["registration-timeout"] <= 0 or cfg["reply-timeout"] <= 0:
        raise ValueError("Timeouts must be positive")
    if cfg["device"] not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be cpu, mps or cuda; Flower assigns each client's GPU")
    if Path(cfg["output"]).suffix != ".json":
        raise ValueError("output must end in .json; model weights use a separate .pt file")
    return cfg
