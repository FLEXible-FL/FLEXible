"""fl_framework_showcase.py - Unified Showcase & Benchmark of Federated Learning Baselines in FLEX.

This script demonstrates and benchmarks the 5 major published FL baselines implemented on FLEX:
  1. FedAvg  - Standard Federated Averaging (McMahan et al., 2017)
  2. FedProx - Proximal Regularization (Li et al., 2020)
  3. FedNova - Normalized Averaging across Heterogeneous Steps (Wang et al., 2020)
  4. FedDyn  - Dynamic Regularization with Gradient Memory (Acar et al., 2021)
  5. MOON    - Model-Contrastive Federated Learning (Li et al., 2021)

Usage:
    # Run a single baseline:
    python examples/fl_framework_showcase.py --method feddyn --rounds 5

    # Run all 5 baselines side-by-side on the exact same non-IID seed:
    python examples/fl_framework_showcase.py --compare-all --rounds 5
"""

from __future__ import annotations

import argparse
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flex.data import Dataset, FedDataset
from flex.model import FlexModel
from flex.pool import (
    FlexPool,
    collect_clients_weights,
    deploy_server_model,
    fed_avg,
    init_server_model,
    set_aggregated_weights,
    weighted_fed_avg,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 1. Model Definition
# ---------------------------------------------------------------------------
class ToyNet(nn.Module):
    """Lightweight 2-layer neural network with a backbone + head (fc) structure."""

    def __init__(self, in_features: int = 16, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# 2. Synthetic Non-IID Dataset Generator
# ---------------------------------------------------------------------------
def make_synthetic_non_iid_data(
    num_clients: int = 4,
    samples_per_client: int = 40,
    test_samples: int = 100,
    in_features: int = 16,
    num_classes: int = 2,
    seed: int = 42,
) -> Tuple[FedDataset, Dataset]:
    """Generates synthetic non-IID feature clusters across clients."""
    gen = torch.Generator().manual_seed(seed)
    class_centers = torch.randn(num_classes, in_features, generator=gen) * 2.0

    client_datasets = {}
    for cid in range(num_clients):
        p_class = cid % num_classes
        labels = []
        features = []
        for _ in range(samples_per_client):
            label = p_class if torch.rand(1, generator=gen).item() < 0.85 else (1 - p_class) % num_classes
            feat = class_centers[label] + torch.randn(in_features, generator=gen) * 0.5
            features.append(feat)
            labels.append(label)
        client_datasets[f"client_{cid}"] = Dataset(features, labels)

    test_features = []
    test_labels = []
    for _ in range(test_samples):
        label = torch.randint(0, num_classes, (1,), generator=gen).item()
        feat = class_centers[label] + torch.randn(in_features, generator=gen) * 0.5
        test_features.append(feat)
        test_labels.append(label)

    test_data = Dataset(test_features, test_labels)
    client_datasets["server"] = test_data
    return FedDataset(client_datasets), test_data


# ---------------------------------------------------------------------------
# 3. FLEX Framework Lifecycle Hooks & Decorators
# ---------------------------------------------------------------------------
@init_server_model
def build_server_model(lr: float = 0.05) -> FlexModel:
    """Initializes server model state inside FlexPool."""
    m = FlexModel()
    m["model"] = ToyNet()
    m["criterion"] = nn.CrossEntropyLoss()
    m["optimizer_func"] = torch.optim.SGD
    m["optimizer_kwargs"] = {"lr": lr}
    return m


@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel) -> FlexModel:
    """Copies server model to clients. Existing client state (e.g. feddyn_grad, prev_model) is retained."""
    client_model = FlexModel()
    client_model["model"] = copy.deepcopy(server_flex_model["model"])
    client_model["server_model"] = copy.deepcopy(server_flex_model["model"])
    client_model["criterion"] = copy.deepcopy(server_flex_model["criterion"])
    client_model["optimizer_func"] = copy.deepcopy(server_flex_model["optimizer_func"])
    client_model["optimizer_kwargs"] = copy.deepcopy(server_flex_model["optimizer_kwargs"])
    return client_model


@collect_clients_weights
def get_clients_weights(client_flex_model: FlexModel) -> List[torch.Tensor]:
    """Extracts weight differences (delta) between local and global model."""
    w_client = client_flex_model["model"].state_dict()
    w_server = client_flex_model["server_model"].state_dict()
    return [(w_client[k] - w_server[k]).float().cpu() for k in w_client]


@set_aggregated_weights
def set_aggregated_weights_to_server(
    server_flex_model: FlexModel, aggregated_weights: List[torch.Tensor]
) -> None:
    """Applies aggregated weight deltas back to the global server model."""
    with torch.no_grad():
        server_sd = server_flex_model["model"].state_dict()
        for k, diff in zip(server_sd, aggregated_weights):
            server_sd[k].add_(diff.to(server_sd[k].device))


# ---------------------------------------------------------------------------
# 4. Method Regularizers & Training Helpers
# ---------------------------------------------------------------------------
def fedprox_loss(model: nn.Module, server_model: nn.Module, mu: float) -> torch.Tensor:
    """FedProx proximal penalty term."""
    prox = torch.tensor(0.0, device=next(model.parameters()).device)
    for p, s_p in zip(model.parameters(), server_model.parameters()):
        prox = prox + torch.sum((p - s_p.to(p.device)) ** 2)
    return (mu / 2.0) * prox


def get_representation(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """MOON hook to extract penultimate features entering fc layer."""
    captured = {}

    def hook(_module, inputs):
        captured["z"] = inputs[0]

    h = model.fc.register_forward_pre_hook(hook)
    try:
        model(x)
    finally:
        h.remove()
    return captured["z"]


def moon_loss(z: torch.Tensor, z_glob: torch.Tensor, z_prev: torch.Tensor, tau: float) -> torch.Tensor:
    """MOON model-contrastive loss."""
    sim_glob = F.cosine_similarity(z, z_glob) / tau
    sim_prev = F.cosine_similarity(z, z_prev) / tau
    logits = torch.stack([sim_glob, sim_prev], dim=1)
    labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


def feddyn_loss(
    model: nn.Module, server_model: nn.Module, prev_grad: Optional[List[torch.Tensor]], alpha: float
) -> torch.Tensor:
    """FedDyn dynamic linear and quadratic regularization."""
    device = next(model.parameters()).device
    linear_pen = torch.tensor(0.0, device=device)
    quad_pen = torch.tensor(0.0, device=device)
    if prev_grad is None:
        prev_grad = [torch.zeros_like(p, device=device) for p in model.parameters()]
    for p, s_p, g in zip(model.parameters(), server_model.parameters(), prev_grad):
        linear_pen = linear_pen + torch.sum(p * g.to(device))
        quad_pen = quad_pen + torch.sum((p - s_p.to(device)) ** 2)
    return -linear_pen + (alpha / 2.0) * quad_pen


@torch.no_grad()
def feddyn_update_grad(
    model: nn.Module, server_model: nn.Module, prev_grad: Optional[List[torch.Tensor]], alpha: float
) -> List[torch.Tensor]:
    """FedDyn local gradient memory update."""
    new_grads = []
    device = next(model.parameters()).device
    if prev_grad is None:
        prev_grad = [torch.zeros_like(p, device=device) for p in model.parameters()]
    for p, s_p, g in zip(model.parameters(), server_model.parameters(), prev_grad):
        updated_g = g.to(device) - alpha * (p - s_p.to(device))
        new_grads.append(updated_g.detach().clone().cpu())
    return new_grads


def train_client(
    client_flex_model: FlexModel,
    client_data: Dataset,
    method: str = "fedavg",
    epochs: int = 3,
    batch_size: int = 8,
    fedprox_mu: float = 0.01,
    feddyn_alpha: float = 0.01,
    moon_mu: float = 1.0,
    moon_tau: float = 0.5,
    device: str = "cpu",
) -> None:
    """Dispatches client training according to the chosen FL baseline."""
    # Simulate step heterogeneity for FedNova
    client_epochs = epochs + (len(client_data) % 3) if method == "fednova" else epochs

    loader = DataLoader(client_data.to_torchvision_dataset(), batch_size=batch_size, shuffle=True)
    model = client_flex_model["model"].to(device).train()
    server_model = client_flex_model["server_model"].to(device).eval()
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    criterion = client_flex_model["criterion"]

    prev_grad = client_flex_model.get("feddyn_grad") if method == "feddyn" else None
    prev_model = client_flex_model.get("prev_model") if method == "moon" else None
    if prev_model is not None:
        prev_model = prev_model.to(device).eval()

    num_iterations = 0
    for _ in range(client_epochs):
        for x, y in loader:
            num_iterations += 1
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            if method == "fedprox" and fedprox_mu > 0.0:
                loss = loss + fedprox_loss(model, server_model, mu=fedprox_mu)
            elif method == "feddyn" and feddyn_alpha > 0.0:
                loss = loss + feddyn_loss(model, server_model, prev_grad, alpha=feddyn_alpha)
            elif method == "moon" and prev_model is not None and moon_mu > 0.0:
                z = get_representation(model, x)
                with torch.no_grad():
                    z_glob = get_representation(server_model, x)
                    z_prev = get_representation(prev_model, x)
                loss = loss + moon_mu * moon_loss(z, z_glob, z_prev, tau=moon_tau)

            loss.backward()
            optimizer.step()

    # Post-training state updates per method
    if method == "fednova":
        client_flex_model["fednova_iters"] = num_iterations
    elif method == "feddyn":
        client_flex_model["feddyn_grad"] = feddyn_update_grad(
            model, server_model, prev_grad, alpha=feddyn_alpha
        )
    elif method == "moon":
        client_flex_model["prev_model"] = copy.deepcopy(model).cpu()


def evaluate_server(
    server_flex_model: FlexModel, test_data: Dataset, device: str = "cpu"
) -> Tuple[float, float]:
    """Computes test loss and accuracy on server's test set."""
    model = server_flex_model["model"]
    loader = DataLoader(test_data.to_torchvision_dataset(), batch_size=32, shuffle=False)
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * len(y)
            pred = out.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += len(y)
    return total_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# 5. Pipeline Runner
# ---------------------------------------------------------------------------
def run_simulation(
    method: str = "fedavg",
    rounds: int = 5,
    clients: int = 4,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 0.05,
    fedprox_mu: float = 0.01,
    feddyn_alpha: float = 0.01,
    moon_mu: float = 1.0,
    moon_tau: float = 0.5,
    seed: int = 42,
    quiet: bool = False,
) -> Tuple[float, float]:
    """Executes a complete federated training simulation for a specific baseline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not quiet:
        print(f"\n--- Running Method: {method.upper()} (Rounds={rounds}, Clients={clients}, Device={device}) ---")

    fed_data, test_data = make_synthetic_non_iid_data(num_clients=clients, seed=seed)
    pool = FlexPool.client_server_pool(fed_data, build_server_model, lr=lr)

    for r in range(rounds):
        # 1. Deploy
        pool.servers.map(copy_server_model_to_clients, pool.clients)

        # 2. Local Training
        pool.clients.map(
            train_client,
            method=method,
            epochs=epochs,
            batch_size=batch_size,
            fedprox_mu=fedprox_mu,
            feddyn_alpha=feddyn_alpha,
            moon_mu=moon_mu,
            moon_tau=moon_tau,
            device=device,
        )

        # 3. Collect
        pool.aggregators.map(get_clients_weights, pool.clients)

        # 4. Aggregate
        if method == "fednova":
            iters = pool.clients.map(lambda cm, _: cm.get("fednova_iters", 1))
            total_iters = sum(iters)
            weights = [it / total_iters for it in iters]
            pool.aggregators.map(weighted_fed_avg, ponderation=weights)
        else:
            pool.aggregators.map(fed_avg)

        # 5. Update server
        pool.aggregators.map(set_aggregated_weights_to_server, pool.servers)

        loss, acc = pool.servers.map(evaluate_server, device=device)[0]
        if not quiet:
            print(f"Round {r + 1:2d}/{rounds:2d} | Server Test Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")

    return loss, acc


def run_benchmark_suite(
    rounds: int = 5, clients: int = 4, epochs: int = 3, seed: int = 42
) -> None:
    """Runs all 5 baselines under the identical seed and dataset partition."""
    methods = ["fedavg", "fedprox", "fednova", "feddyn", "moon"]
    results: Dict[str, Tuple[float, float]] = {}

    print("\n" + "=" * 65)
    print("      FLEX BASELINES COMPARATIVE BENCHMARK SHOWCASE")
    print("=" * 65)

    for m in methods:
        loss, acc = run_simulation(
            method=m,
            rounds=rounds,
            clients=clients,
            epochs=epochs,
            seed=seed,
            quiet=False,
        )
        results[m] = (loss, acc)

    print("\n" + "=" * 65)
    print(f"  FINAL SUMMARY (Evaluated after {rounds} federation rounds)")
    print("=" * 65)
    print(f"| {'Method':<12} | {'Final Test Loss':<18} | {'Final Accuracy':<16} |")
    print("|" + "-" * 14 + "|" + "-" * 20 + "|" + "-" * 18 + "|")
    for m, (loss, acc) in results.items():
        print(f"| {m.upper():<12} | {loss:<18.4f} | {acc * 100:<15.2f}% |")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FLEX FL Baselines Showcase")
    parser.add_argument(
        "--method",
        type=str,
        choices=["fedavg", "fedprox", "fednova", "feddyn", "moon"],
        default="fedavg",
        help="Specific FL baseline to execute",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run all 5 baselines sequentially and display a comparative summary table",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of federation rounds")
    parser.add_argument("--clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--epochs", type=int, default=3, help="Local epochs per client")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--fedprox-mu", type=float, default=0.01, help="FedProx mu")
    parser.add_argument("--feddyn-alpha", type=float, default=0.01, help="FedDyn alpha")
    parser.add_argument("--moon-mu", type=float, default=1.0, help="MOON mu")
    parser.add_argument("--moon-tau", type=float, default=0.5, help="MOON tau")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.compare_all:
        run_benchmark_suite(
            rounds=args.rounds,
            clients=args.clients,
            epochs=args.epochs,
            seed=args.seed,
        )
    else:
        run_simulation(
            method=args.method,
            rounds=args.rounds,
            clients=args.clients,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fedprox_mu=args.fedprox_mu,
            feddyn_alpha=args.feddyn_alpha,
            moon_mu=args.moon_mu,
            moon_tau=args.moon_tau,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
