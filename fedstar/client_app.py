"""Flower Message API client, calling the same local trainer as sequential runs."""

import json
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .train import RunConfig, make_data, local_train

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    config = RunConfig(**json.loads(msg.content["config"]["experiment"]))
    cid = int(context.node_config["partition-id"])
    if int(context.node_config["num-partitions"]) != config.clients:
        raise ValueError("Flower node count must equal config.clients")
    clients, _, classes = make_data(config)
    weights, count, metrics = local_train(
        msg.content["arrays"].to_numpy_ndarrays(),
        clients[cid],
        len(classes),
        config,
        int(msg.content["config"]["server-round"]) - 1,
        cid,
    )
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(weights),
                "metrics": MetricRecord(
                    {"num-examples": count, "client-id": cid, **metrics}
                ),
            }
        ),
        reply_to=msg,
    )
