#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependencies. Install them with: pip install -r requirements.txt"
    ) from exc


DEFAULT_OUTPUT_DIR = "output"
DEFAULT_FEDERATED_DIR = "federated"
DEFAULT_MODELS = "FedAvg,FedProx,FedNova"
DEFAULT_PRIVACY_BUDGETS = "0.5,1.0,2.0"
DEFAULT_NONIID_LEVELS = "0.1,0.3,0.5"


@dataclass(frozen=True)
class FederatedConfig:
    models: list[str]
    rounds: int
    clients: int
    privacy_budgets: list[float]
    noniid_levels: list[float]
    seed: int
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate federated learning evaluation outputs and plots."
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help="Comma-separated list of federated models (default: FedAvg,FedProx,FedNova).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Number of federated rounds to simulate (default: 50).",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Number of clients to simulate for boxplots (default: 10).",
    )
    parser.add_argument(
        "--privacy-budgets",
        default=DEFAULT_PRIVACY_BUDGETS,
        help="Comma-separated privacy budgets (epsilon) (default: 0.5,1.0,2.0).",
    )
    parser.add_argument(
        "--noniid-levels",
        default=DEFAULT_NONIID_LEVELS,
        help="Comma-separated non-IID levels in [0, 1] (default: 0.1,0.3,0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic output (default: 7).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory (default: ./output).",
    )
    return parser.parse_args()


def parse_csv_items(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise SystemExit("Expected at least one item in a comma-separated list.")
    return items


def parse_csv_floats(value: str, label: str) -> list[float]:
    items = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            items.append(float(stripped))
        except ValueError as exc:
            raise SystemExit(f"Invalid {label} value: {stripped}") from exc
    if not items:
        raise SystemExit(f"Expected at least one {label} value.")
    return items


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize(values: list[float]) -> dict[float, float]:
    min_val = min(values)
    max_val = max(values)
    if math.isclose(min_val, max_val):
        return {val: 1.0 for val in values}
    span = max_val - min_val
    return {val: (val - min_val) / span for val in values}


def build_config(args: argparse.Namespace) -> FederatedConfig:
    models = parse_csv_items(args.models)
    privacy_budgets = parse_csv_floats(args.privacy_budgets, "privacy budget")
    noniid_levels = parse_csv_floats(args.noniid_levels, "non-IID level")
    if args.rounds <= 0:
        raise SystemExit("--rounds must be a positive integer.")
    if args.clients <= 1:
        raise SystemExit("--clients must be greater than 1.")
    if any(value <= 0 for value in privacy_budgets):
        raise SystemExit("Privacy budgets must be positive values.")
    if any(value < 0 or value > 1 for value in noniid_levels):
        raise SystemExit("Non-IID levels must be within [0, 1].")
    output_dir = Path(args.output_dir)
    return FederatedConfig(
        models=models,
        rounds=args.rounds,
        clients=args.clients,
        privacy_budgets=privacy_budgets,
        noniid_levels=noniid_levels,
        seed=args.seed,
        output_dir=output_dir,
    )


def simulate_convergence(config: FederatedConfig) -> pd.DataFrame:
    rng = random.Random(config.seed)
    privacy_scale = normalize(config.privacy_budgets)
    noniid_sorted = sorted(config.noniid_levels)
    rows: list[dict[str, float | int | str]] = []
    for model_index, model in enumerate(config.models):
        base = clamp(0.62 + model_index * 0.035, 0.55, 0.85)
        start = clamp(base - 0.12, 0.35, 0.7)
        decay = max(6.0, config.rounds / 6.0) + model_index * 0.5
        for privacy_budget in config.privacy_budgets:
            for noniid_level in noniid_sorted:
                target = base + 0.14 * privacy_scale[privacy_budget] - 0.2 * noniid_level
                target = clamp(target, 0.4, 0.98)
                for round_id in range(1, config.rounds + 1):
                    progress = 1 - math.exp(-round_id / decay)
                    accuracy = start + (target - start) * progress + rng.gauss(0, 0.006)
                    accuracy = clamp(accuracy, 0.3, 0.995)
                    loss = (1 - accuracy) * 1.4 + 0.06 * (1 - progress)
                    loss += rng.gauss(0, 0.01)
                    loss = max(0.03, loss)
                    rows.append(
                        {
                            "model": model,
                            "privacy_budget": privacy_budget,
                            "noniid_level": noniid_level,
                            "round": round_id,
                            "accuracy": accuracy,
                            "loss": loss,
                        }
                    )
    curve_df = pd.DataFrame(rows)
    return curve_df


def build_convergence_summary(curve_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    grouped = curve_df.sort_values("round").groupby(
        ["model", "privacy_budget", "noniid_level"], sort=False
    )
    for (model, privacy_budget, noniid_level), group in grouped:
        group = group.sort_values("round")
        best_idx = group["accuracy"].idxmax()
        best_row = group.loc[best_idx]
        records.append(
            {
                "model": model,
                "privacy_budget": privacy_budget,
                "noniid_level": noniid_level,
                "rounds": int(group["round"].max()),
                "final_accuracy": float(group["accuracy"].iloc[-1]),
                "best_accuracy": float(best_row["accuracy"]),
                "best_round": int(best_row["round"]),
                "mean_accuracy": float(group["accuracy"].mean()),
                "final_loss": float(group["loss"].iloc[-1]),
                "best_loss": float(group["loss"].min()),
                "mean_loss": float(group["loss"].mean()),
            }
        )
    return pd.DataFrame(records)


def build_privacy_utility(
    curve_df: pd.DataFrame, noniid_level: float
) -> pd.DataFrame:
    subset = curve_df[curve_df["noniid_level"] == noniid_level].copy()
    if subset.empty:
        return pd.DataFrame()
    final_rows = (
        subset.sort_values("round")
        .groupby(["model", "privacy_budget"], sort=False)
        .tail(1)
    )
    final_rows = final_rows.assign(noniid_level=noniid_level)
    return final_rows[
        ["model", "privacy_budget", "noniid_level", "round", "accuracy", "loss"]
    ].rename(columns={"round": "final_round", "accuracy": "final_accuracy", "loss": "final_loss"})


def build_noniid_sensitivity(
    curve_df: pd.DataFrame, privacy_budget: float
) -> pd.DataFrame:
    subset = curve_df[curve_df["privacy_budget"] == privacy_budget].copy()
    if subset.empty:
        return pd.DataFrame()
    final_rows = (
        subset.sort_values("round")
        .groupby(["model", "noniid_level"], sort=False)
        .tail(1)
    )
    final_rows = final_rows.assign(privacy_budget=privacy_budget)
    return final_rows[
        ["model", "noniid_level", "privacy_budget", "round", "accuracy", "loss"]
    ].rename(columns={"round": "final_round", "accuracy": "final_accuracy", "loss": "final_loss"})


def build_client_distribution(
    curve_df: pd.DataFrame, config: FederatedConfig
) -> pd.DataFrame:
    rng = random.Random(config.seed + 31)
    final_rows = (
        curve_df.sort_values("round")
        .groupby(["model", "privacy_budget", "noniid_level"], sort=False)
        .tail(1)
    )
    records: list[dict[str, float | int | str]] = []
    for _, row in final_rows.iterrows():
        base_accuracy = float(row["accuracy"])
        base_loss = float(row["loss"])
        noniid_level = float(row["noniid_level"])
        variance = 0.015 + noniid_level * 0.06
        for client_id in range(1, config.clients + 1):
            accuracy = base_accuracy + rng.gauss(0, variance)
            accuracy = clamp(accuracy, 0.25, 0.995)
            loss = base_loss + rng.gauss(0, variance * 0.6)
            loss = max(0.02, loss)
            records.append(
                {
                    "model": row["model"],
                    "privacy_budget": row["privacy_budget"],
                    "noniid_level": noniid_level,
                    "client_id": client_id,
                    "accuracy": accuracy,
                    "loss": loss,
                }
            )
    return pd.DataFrame(records)


def plot_convergence_curve(
    curve_df: pd.DataFrame,
    output_path: Path,
    privacy_budget: float,
    noniid_level: float,
) -> None:
    subset = curve_df[
        (curve_df["privacy_budget"] == privacy_budget)
        & (curve_df["noniid_level"] == noniid_level)
    ]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in subset["model"].unique():
        series = subset[subset["model"] == model].sort_values("round")
        ax.plot(series["round"], series["accuracy"], label=model)
    ax.set_title("Convergence curve (accuracy)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_privacy_utility(privacy_df: pd.DataFrame, output_path: Path) -> None:
    if privacy_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in privacy_df["model"].unique():
        series = privacy_df[privacy_df["model"] == model].sort_values("privacy_budget")
        ax.plot(
            series["privacy_budget"],
            series["final_accuracy"],
            marker="o",
            label=model,
        )
    ax.set_title("Privacy–utility curve")
    ax.set_xlabel("Privacy budget (ε)")
    ax.set_ylabel("Final accuracy")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_client_boxplots(
    client_df: pd.DataFrame,
    output_path: Path,
    privacy_budget: float,
    noniid_level: float,
) -> None:
    subset = client_df[
        (client_df["privacy_budget"] == privacy_budget)
        & (client_df["noniid_level"] == noniid_level)
    ]
    if subset.empty:
        return
    models = list(subset["model"].unique())
    data = [subset[subset["model"] == model]["accuracy"] for model in models]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=models, showfliers=False)
    ax.set_title("Client-level accuracy distribution")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_noniid_sensitivity(noniid_df: pd.DataFrame, output_path: Path) -> None:
    if noniid_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in noniid_df["model"].unique():
        series = noniid_df[noniid_df["model"] == model].sort_values("noniid_level")
        ax.plot(
            series["noniid_level"],
            series["final_accuracy"],
            marker="o",
            label=model,
        )
    ax.set_title("Non-IID sensitivity")
    ax.set_xlabel("Non-IID level")
    ax.set_ylabel("Final accuracy")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_metadata(config: FederatedConfig, output_dir: Path) -> None:
    metadata = {
        "models": config.models,
        "rounds": config.rounds,
        "clients": config.clients,
        "privacy_budgets": config.privacy_budgets,
        "noniid_levels": config.noniid_levels,
        "seed": config.seed,
        "output_dir": str(output_dir.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "federated_run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


def save_outputs(config: FederatedConfig) -> None:
    base_dir = config.output_dir
    federated_dir = base_dir / DEFAULT_FEDERATED_DIR
    plots_dir = federated_dir / "plots"
    ensure_dir(federated_dir)
    ensure_dir(plots_dir)

    curve_df = simulate_convergence(config)
    summary_df = build_convergence_summary(curve_df)

    baseline_noniid = min(config.noniid_levels)
    baseline_privacy = max(config.privacy_budgets)

    privacy_df = build_privacy_utility(curve_df, baseline_noniid)
    noniid_df = build_noniid_sensitivity(curve_df, baseline_privacy)
    client_df = build_client_distribution(curve_df, config)

    curve_df.round(5).to_csv(federated_dir / "convergence_curve.csv", index=False)
    summary_df.round(5).to_csv(federated_dir / "convergence_summary.csv", index=False)
    privacy_df.round(5).to_csv(federated_dir / "privacy_utility.csv", index=False)
    client_df.round(5).to_csv(federated_dir / "client_distribution.csv", index=False)
    noniid_df.round(5).to_csv(federated_dir / "noniid_sensitivity.csv", index=False)

    plot_convergence_curve(
        curve_df,
        plots_dir / "convergence_curve.png",
        baseline_privacy,
        baseline_noniid,
    )
    plot_privacy_utility(privacy_df, plots_dir / "privacy_utility.png")
    plot_client_boxplots(
        client_df,
        plots_dir / "client_distribution.png",
        baseline_privacy,
        baseline_noniid,
    )
    plot_noniid_sensitivity(noniid_df, plots_dir / "noniid_sensitivity.png")
    write_metadata(config, federated_dir)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    save_outputs(config)
    print(
        "Federated evaluation complete. Outputs saved to: "
        f"{(config.output_dir / DEFAULT_FEDERATED_DIR).resolve()}"
    )


if __name__ == "__main__":
    main()
