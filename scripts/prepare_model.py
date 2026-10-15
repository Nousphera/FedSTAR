"""Pretrain the paper encoder on prepared FSD50K development audio."""

import argparse

from fedstar.data import verify_dataset
from fedstar.pretrain import pretrain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="checkpoints/fsd50k")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--resume", action="store_true", help="Resume OUTPUT/checkpoint.pt; epochs is the total target")
    args = parser.parse_args()
    data = verify_dataset("fsd50k", args.data_root)
    pretrain(data.train, args.output, epochs=args.epochs, batch_size=args.batch_size,
             learning_rate=args.learning_rate, seed=args.seed, device=args.device,
             resume=args.resume, dataset_fingerprint=data.fingerprint)
    print(f"Encoder ready: {args.output}/encoder.pt")


if __name__ == "__main__":
    main()
