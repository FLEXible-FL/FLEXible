"""02_fedprox.py - FedProx (Federated Proximal) on the FLEX framework.

Reference:
    Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
    Federated optimization in heterogeneous networks. MLSys 2020.

Key Idea:
    Under statistical and system heterogeneity, local client training drifts away
    from the global model objective. FedProx introduces a proximal regularization
    term to each client's local loss:
        L_prox(theta; theta_t) = L_task(theta) + (mu / 2) * ||theta - theta_t||^2

In FLEX:
    1. Server deploys the global model as both 'model' (trainable) and 'server_model' (reference).
    2. During local_train, the client computes the squared Euclidean distance between its
       current parameters and server_model's parameters, scaling it by mu / 2.
    3. Aggregation uses standard FedAvg (fed_avg).
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
    fed_avg,
    init_server_model,
    set_aggregated_weights,
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
# 2. Synthetic Non-IID Dataset Generator
# ---------------------------------------------------------------------------
def make_synthetic_non_iid_data(
    num_clients: int = 5,
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
# 4. FedProx Local Training Function
# ---------------------------------------------------------------------------
def fedprox_regularization(
    model: nn.Module, server_model: nn.Module, mu: float = 0.01
) -> torch.Tensor:
    """Computes (mu / 2) * sum_p ||p - p_server||^2."""
    prox_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for p, p_server in zip(model.parameters(), server_model.parameters()):
        prox_loss = prox_loss + torch.sum((p - p_server.to(p.device)) ** 2)
    return (mu / 2.0) * prox_loss


def local_train_fedprox(
    client_flex_model: FlexModel,
    client_data: Dataset,
    mu: float = 0.01,
    epochs: int = 3,
    batch_size: int = 8,
    device: str = "cpu",
) -> None:
    """Local client training with FedProx proximal regularization."""
    loader = DataLoader(client_data.to_torchvision_dataset(), batch_size=batch_size, shuffle=True)
    model = client_flex_model["model"].to(device).train()
    server_model = client_flex_model["server_model"].to(device).eval()
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    criterion = client_flex_model["criterion"]

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            task_loss = criterion(out, y)

            # FedProx proximal term
            prox_loss = fedprox_regularization(model, server_model, mu=mu)
            total_loss = task_loss + prox_loss

            total_loss.backward()
            optimizer.step()


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
def run_fedprox(
    rounds: int = 5,
    clients: int = 4,
    mu: float = 0.01,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 0.05,
    seed: int = 42,
) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running FedProx (Rounds: {rounds}, Clients: {clients}, mu: {mu}, Device: {device}) ===")

    fed_data, test_data = make_synthetic_non_iid_data(
        num_clients=clients, seed=seed
    )
    pool = FlexPool.client_server_pool(
        fed_data, build_server_model, lr=lr
    )

    for r in range(rounds):
        # 1. Deploy server model to clients
        pool.servers.map(copy_server_model_to_clients, pool.clients)

        # 2. Local client training with proximal loss
        pool.clients.map(
            local_train_fedprox,
            mu=mu,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )

        # 3. Collect weight updates
        pool.aggregators.map(get_clients_weights, pool.clients)

        # 4. Standard coordinate-wise FedAvg
        pool.aggregators.map(fed_avg)

        # 5. Update global server model
        pool.aggregators.map(set_aggregated_weights_to_server, pool.servers)

        # Evaluate server model
        loss, acc = pool.servers.map(evaluate_server, device=device)[0]
        print(f"Round {r + 1:2d}/{rounds:2d} | Server Test Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")

    print("FedProx completed successfully.\n")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="FedProx Example in FLEX")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federation rounds")
    parser.add_argument("--clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal regularization factor")
    parser.add_argument("--epochs", type=int, default=3, help="Local epochs per client")
    parser.add_argument("--batch-size", type=int, default=8, help="Local batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_fedprox(
        rounds=args.rounds,
        clients=args.clients,
        mu=args.mu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
