"""Native Flower server with sample-weighted aggregation and held-out evaluation."""

import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp
from .config import defaults, validate
from .data import load_data
from .methods import aggregate, partition_records
from .models import build_model
from .training import evaluate, seed_everything

app = ServerApp()


def wait_for_nodes(grid, count, timeout):
    deadline = time.monotonic() + timeout
    while True:
        nodes = sorted(grid.get_node_ids())
        if len(set(nodes)) != len(nodes) or len(nodes) > count:
            raise ValueError("Unexpected Flower SuperNodes")
        if len(nodes) == count:
            return nodes
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Only {len(nodes)}/{count} Flower SuperNodes registered")
        time.sleep(.25)


def replies(grid, messages, timeout):
    expected = {m.metadata.dst_node_id: m for m in messages}
    seen = set()
    for reply in grid.send_and_receive(messages, timeout=timeout):
        node = reply.metadata.src_node_id
        if reply.has_error() or node not in expected or node in seen:
            raise RuntimeError("Each selected client must return one successful update")
        if reply.metadata.group_id != expected[node].metadata.group_id:
            raise ValueError("Reply round mismatch")
        seen.add(node)
        yield reply
    if seen != set(expected):
        raise RuntimeError("Missing client replies; round was not aggregated")


def run_server(grid, cfg):
    cfg = validate(defaults() | cfg)
    seed_everything(cfg["seed"], cfg["torch-threads"])
    data = load_data(cfg["dataset"], cfg["data-root"])
    partitions = partition_records(data.train, cfg)
    model = build_model(len(data.classes))
    encoder_hash = None
    if cfg["pretrained-encoder"]:
        path = Path(cfg["pretrained-encoder"])
        model.encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        encoder_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    count = len(partitions)
    nodes = wait_for_nodes(grid, count, cfg["registration-timeout"])

    def message(node, rnd):
        content = RecordDict({"config": ConfigRecord({"round": rnd, "phase": "train" if rnd else "identify"})})
        if rnd:
            content["model"] = ArrayRecord(model.state_dict())
        return Message(content, dst_node_id=node, message_type="train", group_id=str(rnd))

    mapping = {}
    for reply in replies(grid, [message(node, 0) for node in nodes], cfg["reply-timeout"]):
        cid = reply.content["metrics"]["client-id"]
        if type(cid) != int or cid in mapping or not 0 <= cid < count:
            raise ValueError("Client partition IDs must be unique and complete")
        mapping[cid] = reply.metadata.src_node_id
    rng = np.random.default_rng(cfg["seed"])
    history = []
    for rnd in range(1, cfg["num-rounds"] + 1):
        selected = sorted(rng.choice(count, max(1, math.floor(count * cfg["participation"])), replace=False).tolist())
        updates, weights, metrics = [], [], []
        for reply in replies(grid, [message(mapping[cid], rnd) for cid in selected], cfg["reply-timeout"]):
            metric = dict(reply.content["metrics"])
            cid, n = metric["client-id"], metric["num-examples"]
            if (cid not in selected or mapping[cid] != reply.metadata.src_node_id
                    or type(n) != int or n != len(partitions[cid]) or not all(math.isfinite(v) for v in metric.values())):
                raise ValueError("Invalid client metrics or sample count")
            updates.append(reply.content["model"].to_torch_state_dict())
            weights.append(n)
            metrics.append(metric)
        model.load_state_dict(aggregate(updates, weights, model.state_dict()), strict=True)
        evaluation = evaluate(model, data.test, cfg)
        history.append({"round": rnd, "clients": metrics,
                        "train-loss": sum(m["loss"] * m["num-examples"] for m in metrics) / sum(weights), **evaluation})
        print(json.dumps({"round": rnd, **evaluation}), flush=True)
    ownership = [{"labeled": [r["id"] for r in rows if r["label"] >= 0],
                  "unlabeled": [r["id"] for r in rows if r["label"] < 0]} for rows in partitions]
    return model, {
        "schema": 1, "config": cfg, "classes": data.classes, "data-fingerprint": data.fingerprint,
        "pretrained-sha256": encoder_hash, "partitions": ownership, "rounds": history,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "flower": version("flwr")},
    }


@app.main()
def main(grid: Grid, context: Context):
    cfg = validate(defaults() | context.run_config)
    path = Path(cfg["output"])
    checkpoint = path.with_suffix(".pt")
    if path.exists() or checkpoint.exists():
        raise FileExistsError("Output exists; choose a new --output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    model, result = run_server(grid, cfg)
    with checkpoint.open("xb") as stream:
        torch.save({"model": model.state_dict(), "classes": result["classes"], "config": cfg}, stream)
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
