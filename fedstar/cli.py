"""No import-time downloads or dataset access."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="FedSTAR data preparation and training"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--task",
        choices=["speech_commands", "voxforge", "ambient_context"],
        required=True,
    )
    for arg in ("train-root", "test-root", "train-split", "test-split", "output"):
        prepare.add_argument("--" + arg, required=True, type=Path)
    for name in ("train", "flower"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/smoke.json")
        command.add_argument("--output")
        command.add_argument("--seed", type=int)
    pretrain = commands.add_parser("pretrain")
    pretrain.add_argument(
        "--clips",
        type=Path,
        required=True,
        help="JSON list of local FSD50K WAV paths (no labels)",
    )
    pretrain.add_argument("--output", required=True)
    pretrain.add_argument("--epochs", type=int, default=500)
    pretrain.add_argument("--batch-size", type=int, default=1024)
    pretrain.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    if args.command == "prepare":
        from .data import read_splits

        rows, classes = read_splits(
            args.task,
            args.train_root,
            args.test_root,
            args.train_split,
            args.test_split,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(
                {"schema": 1, "task": args.task, "classes": classes, "records": rows},
                f,
                indent=2,
            )
        print(f"Indexed {len(rows)} clips with {len(classes)} classes")
    elif args.command == "pretrain":
        from .pretrain import pretrain

        pretrain(
            json.loads(args.clips.read_text()),
            args.output,
            args.epochs,
            args.batch_size,
            args.seed,
        )
    else:
        from .train import load_config, run

        config = load_config(args.config, output=args.output, seed=args.seed)
        if args.command == "train":
            run(config)
        else:
            from .server_app import simulate

            simulate(config)


if __name__ == "__main__":
    main()
