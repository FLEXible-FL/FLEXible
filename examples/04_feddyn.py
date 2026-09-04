"""04_feddyn.py - FedDyn (Federated Dynamic Regularization) on the FLEX framework.

Reference:
    Acar, D. A. E., Zhao, Y., Navarro, R. M., Mattina, M., Whatmough, P. N., & Saligrama, V. (2021).
    Federated learning based on dynamic regularization. ICLR 2021.

Key Idea:
    In non-IID settings, client empirical risk minimizers diverge from the global
    objective. FedDyn provably guarantees convergence to the global stationary points
    by augmenting each client's local loss with a dynamic regularization term composed of:
      1. A linear term matching historical local gradient drift: - <h_k^t, theta>
      2. A quadratic proximal term penalizing divergence from the global model: (alpha / 2) * ||theta - theta^t||^2
    Each client maintains a persistent local gradient memory vector h_k updated after every round:
      h_k^{t+1} = h_k^t - alpha * (theta_k - theta^t)

In FLEX:
    1. Persistent Client State: Each client stores h_k in client_flex_model['feddyn_grad'].
       FLEX's deploy hook updates model parameters while preserving client-specific state.
    2. Dynamic Regularization: During local_train, the client computes the linear and quadratic
       dynamic penalties and adds them to task loss.
    3. State Update: At the end of local training, the client updates its gradient memory h_k^{t+1}.
    4. Aggregation: Standard parameter delta averaging via flex.pool.fed_avg.
"""

from __future__ import annotations

import argparse
import copy
from typing import List, Optional, Tuple

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
    """Copies server model to clients. Existing client state (e.g. feddyn_grad) is retained."""
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
# 4. FedDyn Dynamic Regularization & Gradient Memory
# ---------------------------------------------------------------------------
def compute_feddyn_loss(
    model: nn.Module,
    server_model: nn.Module,
    prev_grad: Optional[List[torch.Tensor]],
    alpha: float,
) -> torch.Tensor:
    """Computes dynamic linear and quadratic regularization penalty terms."""
    device = next(model.parameters()).device
    linear_penalty = torch.tensor(0.0, device=device)
    quad_penalty = torch.tensor(0.0, device=device)

    if prev_grad is None:
        prev_grad = [torch.zeros_like(p, device=device) for p in model.parameters()]

    for p, s_p, g in zip(model.parameters(), server_model.parameters(), prev_grad):
        # Linear term: - <h_k, theta>
        linear_penalty = linear_penalty + torch.sum(p * g.to(device))
        # Quadratic term: (alpha / 2) * ||theta - theta_glob||^2
        quad_penalty = quad_penalty + torch.sum((p - s_p.to(device)) ** 2)

    return -linear_penalty + (alpha / 2.0) * quad_penalty


@torch.no_grad()
def update_local_grad_state(
    model: nn.Module,
    server_model: nn.Module,
    prev_grad: Optional[List[torch.Tensor]],
    alpha: float,
) -> List[torch.Tensor]:
    """Updates client gradient memory: h_k^{t+1} = h_k^t - alpha * (theta_k - theta^t)."""
    new_grad_state = []
    device = next(model.parameters()).device
    if prev_grad is None:
        prev_grad = [torch.zeros_like(p, device=device) for p in model.parameters()]

    for p, s_p, g in zip(model.parameters(), server_model.parameters(), prev_grad):
        updated_g = g.to(device) - alpha * (p - s_p.to(device))
        new_grad_state.append(updated_g.detach().clone().cpu())

    return new_grad_state


def local_train_feddyn(
    client_flex_model: FlexModel,
    client_data: Dataset,
    alpha: float = 0.01,
    epochs: int = 3,
    batch_size: int = 8,
    device: str = "cpu",
) -> None:
    """Client training with FedDyn dynamic regularization."""
    loader = DataLoader(client_data.to_torchvision_dataset(), batch_size=batch_size, shuffle=True)
    model = client_flex_model["model"].to(device).train()
    server_model = client_flex_model["server_model"].to(device).eval()
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    criterion = client_flex_model["criterion"]
    prev_grad = client_flex_model.get("feddyn_grad")

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            task_loss = criterion(out, y)

            # FedDyn dynamic regularizer
            dyn_loss = compute_feddyn_loss(model, server_model, prev_grad, alpha=alpha)
            total_loss = task_loss + dyn_loss

            total_loss.backward()
            optimizer.step()

    # Update and persist client gradient state across rounds
    client_flex_model["feddyn_grad"] = update_local_grad_state(
        model, server_model, prev_grad, alpha=alpha
    )


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
def run_feddyn(
    rounds: int = 5,
    clients: int = 4,
    alpha: float = 0.01,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 0.05,
    seed: int = 42,
) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running FedDyn (Rounds: {rounds}, Clients: {clients}, alpha: {alpha}, Device: {device}) ===")

    fed_data, test_data = make_synthetic_non_iid_data(
        num_clients=clients, seed=seed
    )
    pool = FlexPool.client_server_pool(
        fed_data, build_server_model, lr=lr
    )

    for r in range(rounds):
        # 1. Deploy server model to clients (feddyn_grad is preserved)
        pool.servers.map(copy_server_model_to_clients, pool.clients)

        # 2. Local training with dynamic regularizer
        pool.clients.map(
            local_train_feddyn,
            alpha=alpha,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )

        # 3. Collect client weight deltas
        pool.aggregators.map(get_clients_weights, pool.clients)

        # 4. FedAvg aggregation of updates
        pool.aggregators.map(fed_avg)

        # 5. Update global server model
        pool.aggregators.map(set_aggregated_weights_to_server, pool.servers)

        # Evaluate server model
        loss, acc = pool.servers.map(evaluate_server, device=device)[0]
        print(f"Round {r + 1:2d}/{rounds:2d} | Server Test Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")

    print("FedDyn completed successfully.\n")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="FedDyn Example in FLEX")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federation rounds")
    parser.add_argument("--clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--alpha", type=float, default=0.01, help="FedDyn dynamic regularization alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Local epochs per client")
    parser.add_argument("--batch-size", type=int, default=8, help="Local batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_feddyn(
        rounds=args.rounds,
        clients=args.clients,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
