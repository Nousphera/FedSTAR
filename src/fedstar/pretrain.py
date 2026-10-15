"""Centralized FSD50K contrastive pretraining and resumable encoder export."""

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn

from .audio import features, read_wave
from .models import build_encoder, contrastive_loss, regularization_loss


class ContrastiveModel(nn.Module):
    """Paper encoder and 256-unit COLA bilinear projection (reference [35])."""

    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.projection = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256, eps=1e-3), nn.Tanh())
        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)
        self.bilinear = nn.Parameter(torch.empty(256, 256))
        nn.init.normal_(self.bilinear, std=0.05)

    def forward(self, x):
        return self.projection(self.encoder(x))


def _device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda or mps")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return device


def _save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _signature(records, fingerprint):
    digest = hashlib.sha256()
    digest.update(str(fingerprint or "").encode())
    for record in records:
        path = Path(record["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(json.dumps([record["id"], str(path)]).encode())
        if fingerprint is None:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _batches(order, batch_size):
    start = 0
    while start < len(order):
        end = min(start + batch_size, len(order))
        # A singleton has no negatives: merge it into the preceding batch.
        if len(order) - end == 1:
            end += 1
        yield order[start:end]
        start = end


def pretrain(records, output, epochs=500, batch_size=1024, seed=2021,
             learning_rate=0.001, device="auto", resume=False,
             dataset_fingerprint=None):
    """Train on clip records and export encoder.pt plus a restart checkpoint.

    Two independently sampled crops from each clip form the positive pair.
    Other clips' second crops are negatives. A final singleton is merged into
    the preceding batch, so every clip contributes and the maximum batch is
    batch_size + 1. ``epochs`` is the total target, including resumed epochs.
    ``dataset_fingerprint`` should come from a verified prepared dataset.
    """
    records = list(records)
    if len(records) < 2 or batch_size < 2 or epochs < 1:
        raise ValueError("Pretraining requires at least two clips, batch_size >= 2 and positive epochs")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if len({str(r["id"]) for r in records}) != len(records) or len({str(Path(r["path"]).resolve()) for r in records}) != len(records):
        raise ValueError("Each contrastive record must identify a different audio clip")
    selected_device = _device(device)
    config = {
        "objective": "cola-bilinear-infonce", "batch_size": batch_size,
        "seed": seed, "learning_rate": learning_rate, "adam_epsilon": 1e-7,
        "clips": len(records), "data_signature": _signature(records, dataset_fingerprint),
        "dataset_fingerprint": dataset_fingerprint, "device": str(selected_device),
    }
    output = Path(output)
    checkpoint = None
    if resume:
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("Resume requires identical data, seed, batch size, learning rate and device")
        if epochs < checkpoint["epoch"]:
            raise ValueError("epochs cannot precede the saved checkpoint epoch")
    else:
        output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ContrastiveModel().to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    history = []
    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, start_epoch = checkpoint["history"], checkpoint["epoch"]
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if selected_device.type == "mps":
            torch.mps.set_rng_state(checkpoint["device_rng"])
        elif selected_device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["device_rng"])
    _json(config | {"target_epochs": epochs}, output / "config.json")
    _json({"python": platform.python_version(), "torch": str(torch.__version__),
           "numpy": np.__version__, "platform": platform.platform()}, output / "environment.json")
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss, examples, steps = 0.0, 0, 0
        for indices in _batches(rng.permutation(len(records)), batch_size):
            waves = [read_wave(records[int(i)]["path"]) for i in indices]
            first = torch.stack([features(w, True, rng) for w in waves])
            second = torch.stack([features(w, True, rng) for w in waves])
            optimizer.zero_grad(set_to_none=True)
            a, b = model(torch.cat((first, second)).to(selected_device)).chunk(2)
            loss = contrastive_loss(a, b, model.bilinear) + regularization_loss(model.encoder)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite contrastive loss; checkpoint retains the last completed epoch")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite contrastive gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            examples += len(indices)
            steps += 1
        history.append({"epoch": epoch + 1, "loss": total_loss / examples, "examples": examples, "steps": steps})
        device_rng = None
        if selected_device.type == "mps":
            device_rng = torch.mps.get_rng_state()
        elif selected_device.type == "cuda":
            device_rng = torch.cuda.get_rng_state_all()
        _save({"format_version": 1, "epoch": epoch + 1, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "history": history, "config": config,
               "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
               "device_rng": device_rng}, output / "checkpoint.pt")
        _save({key: value.detach().cpu() for key, value in model.encoder.state_dict().items()}, output / "encoder.pt")
        _json(history, output / "metrics.json")
        print(f"Epoch {epoch + 1}/{epochs}: loss={total_loss / examples:.6f}, clips={examples}", flush=True)
    # Also refresh the export when resuming an already completed checkpoint.
    _save({key: value.detach().cpu() for key, value in model.encoder.state_dict().items()}, output / "encoder.pt")
    return model.encoder.cpu().eval()
