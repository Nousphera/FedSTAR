"""Real Flower ServerApp and weighted FedAvg, aborting incomplete rounds."""

from dataclasses import asdict
import json
import time
import numpy as np
import tensorflow as tf
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from .train import evaluate, initial_model, load_config, make_data, prepare_output


class SeededFedAvg(FedAvg):
    def __init__(self, config):
        super().__init__(
            fraction_train=config.participation,
            fraction_evaluate=0.0,
            min_train_nodes=1,
            min_evaluate_nodes=0,
            min_available_nodes=config.clients,
        )
        self.experiment = config
        self.client_history = {}

    def configure_train(self, server_round, arrays, config, grid):
        nodes = sorted(grid.get_node_ids())
        if len(nodes) != self.experiment.clients:
            raise ValueError("All configured clients must be available")
        rng = np.random.default_rng(
            np.random.SeedSequence([self.experiment.seed, server_round])
        )
        count = max(1, int(len(nodes) * self.experiment.participation))
        selected = [
            nodes[i]
            for i in sorted(rng.choice(len(nodes), count, replace=False).tolist())
        ]
        config["server-round"] = server_round
        return self._construct_messages(
            RecordDict({"arrays": arrays, "config": config}),
            selected,
            MessageType.TRAIN,
        )

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        expected = max(1, int(self.experiment.clients * self.experiment.participation))
        if len(replies) != expected or any(r.has_error() for r in replies):
            raise RuntimeError("Incomplete Flower round")
        self.client_history[server_round] = [
            {"node_id": r.metadata.src_node_id, **dict(r.content["metrics"])}
            for r in replies
        ]
        return super().aggregate_train(server_round, replies)


def create_app(config_override=None):
    application = ServerApp()

    @application.main()
    def main(grid: Grid, context: Context):
        config = config_override or load_config(str(context.run_config["config"]))
        tf.keras.utils.set_random_seed(config.seed)
        tf.config.experimental.enable_op_determinism()
        clients, test, classes = make_data(config)
        output = prepare_output(config, clients, test, classes, "flower-message-api")
        model = initial_model(config, len(classes))
        history = []
        strategy = SeededFedAvg(config)
        start = time.monotonic()

        def evaluate_global(server_round, arrays):
            model.set_weights(arrays.to_numpy_ndarrays())
            metrics = evaluate(model, test)
            history.append(
                {
                    "round": server_round,
                    **metrics,
                    "client_metrics": strategy.client_history.get(server_round, []),
                }
            )
            return MetricRecord(metrics)

        result = strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(model.get_weights()),
            num_rounds=config.rounds,
            train_config=ConfigRecord({"experiment": json.dumps(asdict(config))}),
            evaluate_fn=evaluate_global,
        )
        model.set_weights(result.arrays.to_numpy_ndarrays())
        model.save(output / "model.keras")
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "benchmark_reproduction": False,
                    "dataset": config.dataset,
                    "mode": config.mode,
                    "seconds": time.monotonic() - start,
                    "rounds": history,
                },
                indent=2,
            )
            + "\n"
        )

    return application


app = create_app()


def simulate(config):
    from flwr.simulation import run_simulation
    from .client_app import app as client_app

    run_simulation(
        server_app=create_app(config),
        client_app=client_app,
        num_supernodes=config.clients,
        backend_config={
            "init_args": {
                "num_cpus": min(config.clients, 2),
                "include_dashboard": False,
            },
            "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
        },
    )
