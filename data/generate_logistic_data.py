from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().with_name("synthetic_logistic.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic binary logistic-regression data with labels in {0, 1}.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        required=True,
        help="Feature dimension.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        nargs="+",
        required=True,
        help="Coefficient vector. Its length must match --dimension.",
    )
    parser.add_argument(
        "--intercept",
        type=float,
        default=0.0,
        help="Intercept added to each logit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--flip-prob",
        type=float,
        default=0.0,
        help="Probability of independently flipping a generated label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    return parser


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.n_samples <= 0:
        parser.error("--n-samples must be a positive integer.")
    if args.dimension <= 0:
        parser.error("--dimension must be a positive integer.")
    if len(args.beta) != args.dimension:
        parser.error("--beta length must match --dimension.")
    if not 0.0 <= args.flip_prob < 1.0:
        parser.error("--flip-prob must satisfy 0 <= --flip-prob < 1.")


def generate_dataset(
    *,
    n_samples: int,
    dimension: int,
    beta: np.ndarray,
    intercept: float,
    flip_prob: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.standard_normal(size=(n_samples, dimension))
    logits = features @ beta + intercept
    probabilities = sigmoid(logits)
    labels = rng.binomial(1, probabilities).astype(int)

    if flip_prob > 0.0:
        flip_mask = rng.random(n_samples) < flip_prob
        labels[flip_mask] = 1 - labels[flip_mask]

    return features, labels


def write_csv(output_path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"x{i}" for i in range(1, features.shape[1] + 1)] + ["y"]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row, label in zip(features, labels):
            writer.writerow([f"{value:.18e}" for value in row] + [int(label)])


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    beta = np.asarray(args.beta, dtype=float)
    features, labels = generate_dataset(
        n_samples=args.n_samples,
        dimension=args.dimension,
        beta=beta,
        intercept=float(args.intercept),
        flip_prob=float(args.flip_prob),
        seed=int(args.seed),
    )
    output_path = args.output.resolve()
    write_csv(output_path, features, labels)

    positive_rate = float(labels.mean()) if len(labels) else 0.0
    beta_repr = np.array2string(beta, precision=4, separator=", ")
    print(f"Saved synthetic logistic data to {output_path}")
    print(
        "Summary: "
        f"n_samples={args.n_samples}, "
        f"dimension={args.dimension}, "
        f"beta={beta_repr}, "
        f"intercept={args.intercept:.6f}, "
        f"seed={args.seed}, "
        f"positive_rate={positive_rate:.4f}"
    )


if __name__ == "__main__":
    main()
