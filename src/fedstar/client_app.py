"""Native Flower ClientApp for self-training and supervised baselines."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .config import defaults, validate
from .data import load_data
from .methods import partition_records
from .models import build_model
from .training import seed_everything, train_local

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = validate(defaults() | context.run_config)
    cid = context.node_config["partition-id"]
    count = 1 if cfg["method"] == "centralized" else cfg["num-clients"]
    if type(cid) != int or not 0 <= cid < count:
        raise ValueError("Invalid client partition")
    incoming = msg.content["config"]
    rnd, phase = incoming["round"], incoming["phase"]
    if type(rnd) != int or not 0 <= rnd <= cfg["num-rounds"] or phase not in {"identify", "train"}:
        raise ValueError("Invalid experiment message")
    if (phase == "identify") != (rnd == 0):
        raise ValueError("Invalid initialization round")
    if phase == "identify":
        return Message(RecordDict({"metrics": MetricRecord({"client-id": cid})}), reply_to=msg)
    seed_everything(cfg["seed"] + cid, cfg["torch-threads"])
    data = load_data(cfg["dataset"], cfg["data-root"])
    records = partition_records(data.train, cfg)[cid]
    model = build_model(len(data.classes))
    state = msg.content["model"].to_torch_state_dict()
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Received nonfinite model")
    model.load_state_dict(state, strict=True)
    optimizer = context.state["optimizer"].to_torch_state_dict() if "optimizer" in context.state else None
    metrics, optimizer = train_local(model, records, cfg, cfg["seed"] + rnd * count + cid, rnd, optimizer)
    if cfg["method"] == "centralized":
        context.state["optimizer"] = ArrayRecord(optimizer)
    return Message(RecordDict({
        "model": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"client-id": cid, "num-examples": len(records), **metrics}),
    }), reply_to=msg)
