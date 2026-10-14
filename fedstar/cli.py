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
    validate = commands.add_parser("validate", help="Decode and verify a prepared manifest")
    validate.add_argument("--manifest", required=True, type=Path)
    download = commands.add_parser("download-speech", help="Download official Speech Commands v2 archives (~2.5 GB)")
    download.add_argument("--output", required=True, type=Path)
    speech = commands.add_parser("speech-commands", help="Prepare official archives with a recorded new silence recipe")
    speech.add_argument("--train-root", required=True, type=Path)
    speech.add_argument("--test-root", required=True, type=Path)
    speech.add_argument("--output", required=True, type=Path)
    speech.add_argument("--seed", type=int, default=2021)
    speech.add_argument("--silence-count", type=int, default=668)
    clips = commands.add_parser("prepare-pretraining", help="Validate local FSD50K development WAVs for pretraining")
    clips.add_argument("--root", required=True, type=Path)
    clips.add_argument("--output", required=True, type=Path)
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
        from .preparation import check_records, write_json
        from .data import read_wave
        rows = check_records(rows, classes, ("train", "test"), decode=read_wave)
        write_json(args.output, {"schema": 1, "task": args.task, "classes": classes, "records": rows})
        print(f"Validated {len(rows)} clips with {len(classes)} classes")
    elif args.command == "validate":
        from .preparation import check_records
        from .data import read_wave
        value = json.loads(args.manifest.read_text())
        check_records(value["records"], value["classes"], ("train", "test"), decode=read_wave, verify_hashes=True)
        print(f"Validated {len(value['records'])} clips")
    elif args.command == "download-speech":
        from .datasets import download_speech
        download_speech(args.output)
        print(f"Downloaded official archives to {args.output}")
    elif args.command == "speech-commands":
        from .datasets import prepare_speech
        print(prepare_speech(args.train_root, args.test_root, args.output, args.seed, args.silence_count))
    elif args.command == "prepare-pretraining":
        from .data import read_wave
        from .preparation import write_json
        paths = sorted(p.resolve() for p in args.root.rglob("*.wav"))
        if len(paths) < 2 or len(set(paths)) != len(paths):
            raise ValueError("Need at least two distinct FSD50K development WAV files")
        for path in paths:
            if not path.is_relative_to(args.root.resolve()):
                raise ValueError("Audio path escapes development root")
            read_wave(path)
        write_json(args.output, [str(p) for p in paths])
        print(f"Validated {len(paths)} pretraining clips")
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
