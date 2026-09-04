"""03_fednova.py - FedNova (Federated Normalized Averaging) on the FLEX framework.

Reference:
    Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V. (2020).
    Tackling the objective inconsistency problem in heterogeneous federated optimization.
    NeurIPS 2020.

Key Idea:
    When clients perform different numbers of local solver steps (due to variable
    epochs, unequal dataset sizes, or straggler cut-offs), standard FedAvg suffers
    from "objective inconsistency" - the global model converges towards an unintended
    surrogate objective biased towards clients taking more steps.
    FedNova tracks local iteration counts (tau_k) and weights each client update
    proportionately to normalize the effective gradient progress.

In FLEX:
    1. Clients record their total local iterations in client_flex_model['fednova_iters'].
    2. The server extracts the iteration counts across all participating clients.
    3. Iteration counts are normalized into aggregation weights (e.g. tau_k / sum(tau_j)).
    4. Aggregator applies weighted aggregation via flex.pool.weighted_fed_avg:
       pool.aggregators.map(weighted_fed_avg, ponderation=ponderation_list).
    5. Server updates parameters via @set_aggregated_weights.
"""

from __future__ import annotations

import argparse
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flex.data import Dataset, FedDataset
from flex.model import FlexModel
from flex.pool import (
    FlexPool,
    collect_clients_weights,
    deploy_server_model,
    init_server_model,
    set_aggregated_weights,
    weighted_fed_avg,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 1. Model Definition
# ---------------------------------------------------------------------------
class ToyNet(nn.Module):
    """Lightweight 2-layer neural network for fast demonstration."""

    def __init__(self, in_features: int = 16, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# 2. Synthetic Heterogeneous Dataset Generator
# ---------------------------------------------------------------------------
def make_synthetic_heterogeneous_data(
    num_clients: int = 4,
    min_samples: int = 20,
    max_samples: int = 60,
    test_samples: int = 100,
    in_features: int = 16,
    num_classes: int = 2,
    seed: int = 42,
) -> Tuple[FedDataset, Dataset]:
    """Generates synthetic clients with unequal dataset sizes (data quantity heterogeneity)."""
    gen = torch.Generator().manual_seed(seed)
    class_centers = torch.randn(num_classes, in_features, generator=gen) * 2.0

    client_datasets = {}
    for cid in range(num_clients):
        # Vary sample count per client to simulate system & data heterogeneity
        sample_count = min_samples + int(cid * (max_samples - min_samples) / max(num_clients - 1, 1))
        p_class = cid % num_classes
        labels = []
        features = []
        for _ in range(sample_count):
            label = p_class if torch.rand(1, generator=gen).item() < 0.8 else (1 - p_class) % num_classes
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
    """Copies server model and optimizer settings to participating clients."""
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
# 4. FedNova Local Training with Iteration Tracking
# ---------------------------------------------------------------------------
def local_train_fednova(
    client_flex_model: FlexModel,
    client_data: Dataset,
    base_epochs: int = 2,
    batch_size: int = 8,
    device: str = "cpu",
) -> None:
    """Client training that tracks iteration count tau_k across heterogeneous local steps."""
    # Clients naturally differ in iterations due to unequal dataset sizes and varying local epochs
    client_epochs = base_epochs + (len(client_data) % 3)
    loader = DataLoader(client_data.to_torchvision_dataset(), batch_size=batch_size, shuffle=True)
    model = client_flex_model["model"].to(device).train()
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    criterion = client_flex_model["criterion"]

    num_iterations = 0
    for _ in range(client_epochs):
        for x, y in loader:
            num_iterations += 1
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Crucial FedNova step: store client's local step count
    client_flex_model["fednova_iters"] = num_iterations


def get_fednova_iters(client_flex_model: FlexModel, _data: Dataset) -> int:
    """Extracts the recorded iteration count from client model."""
    return client_flex_model.get("fednova_iters", 1)


def compute_fednova_weights(iteration_counts: List[int]) -> List[float]:
    """Computes normalized weights proportional to each client's iteration count."""
    total_iters = sum(iteration_counts)
    if total_iters == 0:
        return [1.0 / len(iteration_counts)] * len(iteration_counts)
    return [c / total_iters for c in iteration_counts]


def evaluate_server(
    server_flex_model: FlexModel, test_data: Dataset, device: str = "cpu"
) -> Tuple[float, float]:
    """Computes test loss and accuracy on the server's test set."""
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
# 5. Main Federation Loop
# ---------------------------------------------------------------------------
def run_fednova(
    rounds: int = 5,
    clients: int = 4,
    base_epochs: int = 2,
    batch_size: int = 8,
    lr: float = 0.05,
    seed: int = 42,
) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running FedNova (Rounds: {rounds}, Clients: {clients}, Device: {device}) ===")

    fed_data, test_data = make_synthetic_heterogeneous_data(
        num_clients=clients, seed=seed
    )
    pool = FlexPool.client_server_pool(
        fed_data, build_server_model, lr=lr
    )

    for r in range(rounds):
        # 1. Deploy server model to clients
        pool.servers.map(copy_server_model_to_clients, pool.clients)

        # 2. Local client training with variable steps & iteration recording
        pool.clients.map(
            local_train_fednova,
            base_epochs=base_epochs,
            batch_size=batch_size,
            device=device,
        )

        # 3. Collect client weight updates
        pool.aggregators.map(get_clients_weights, pool.clients)

        # 4. FedNova normalized aggregation
        client_iters = pool.clients.map(get_fednova_iters)
        ponderation_weights = compute_fednova_weights(client_iters)

        pool.aggregators.map(weighted_fed_avg, ponderation=ponderation_weights)

        # 5. Update global server model
        pool.aggregators.map(set_aggregated_weights_to_server, pool.servers)

        # Evaluate server model
        loss, acc = pool.servers.map(evaluate_server, device=device)[0]
        iters_str = ", ".join(f"C{i}:{it}" for i, it in enumerate(client_iters))
        print(f"Round {r + 1:2d}/{rounds:2d} | Iters: [{iters_str}] | Server Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")

    print("FedNova completed successfully.\n")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="FedNova Example in FLEX")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federation rounds")
    parser.add_argument("--clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--base-epochs", type=int, default=2, help="Base local epochs per client")
    parser.add_argument("--batch-size", type=int, default=8, help="Local batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_fednova(
        rounds=args.rounds,
        clients=args.clients,
        base_epochs=args.base_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
