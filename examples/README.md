# Federated Learning Baselines on the FLEX Framework

This directory provides deliverable, self-contained single-file reference implementations of the 5 major published Federated Learning (FL) baselines built on top of the [FLEX](https://github.com/FLEXible-FL/FLEX-framework) (`flexible-fl`) framework:

1. **[FedAvg](01_fedavg.py)**: Federated Averaging ([McMahan et al., AISTATS 2017](https://proceedings.mlr.press/v54/mcmahan17a.html))
2. **[FedProx](02_fedprox.py)**: Federated Proximal Regularization ([Li et al., MLSys 2020](https://proceedings.mlsys.org/paper/2020/file/38464ba93822bc99f5722f6a9b00766e-Paper.pdf))
3. **[FedNova](03_fednova.py)**: Normalized Averaging with Heterogeneous Steps ([Wang et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/dedb62c47c14d88a74f333452d69a572-Paper.pdf))
4. **[FedDyn](04_feddyn.py)**: Dynamic Regularization with Client Gradient Memory ([Acar et al., ICLR 2021](https://openreview.net/forum?id=B7v4QMR6Z9w))
5. **[MOON](05_moon.py)**: Model-Contrastive Federated Learning ([Li et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.html))
6. **[Showcase Runner](fl_framework_showcase.py)**: Unified CLI runner and benchmark suite to compare all baselines side-by-side.

All examples run completely out-of-the-box on either **CPU or CUDA** using lightweight in-memory synthetic non-IID data without requiring external downloads.

---

## Method Enumeration & Comparison

| Baseline | Category | Key Mathematical Formulation | Primary Problem Addressed | FLEX Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **FedAvg** | Parameter Aggregation | $\Delta \theta = \frac{1}{K}\sum_{k=1}^K \Delta \theta_k$ | Foundational baseline | Standard `flex.pool.fed_avg` |
| **FedProx** | Client Objective Regularizer | $\mathcal{L} + \frac{\mu}{2}\|\theta - \theta_{\text{server}}\|^2$ | Client drift under statistical heterogeneity | Compares local `model` with `server_model` in local training |
| **FedNova** | Weighted Aggregation | $\Delta \theta = \sum_{k=1}^K \frac{\tau_k}{\sum_j \tau_j} \Delta \theta_k$ | Objective inconsistency from variable local iterations $\tau_k$ | Records `client_model['fednova_iters']` and aggregates via `weighted_fed_avg` |
| **FedDyn** | Dynamic Regularizer | $\mathcal{L} - \langle h_k^t, \theta \rangle + \frac{\alpha}{2}\|\theta - \theta^t\|^2$ | Provable convergence to global stationary points | Persistent memory `client_model['feddyn_grad']` preserved across rounds |
| **MOON** | Representation Regularizer | $\mathcal{L} + \mu \cdot \mathcal{L}_{\text{con}}(z, z_{\text{glob}}, z_{\text{prev}})$ | Representation drift in latent feature space | Forward hook on classifier head + persistent `client_model['prev_model']` |

---

## Architectural Implementation Patterns in FLEX

The FLEX framework uses an actor-pool model (`FlexPool`) managing clients, servers, and aggregators. Extending FLEX to support these methods involves 4 core patterns:

### Pattern 1: Accessing Global Model during Local Training (FedProx)
When `@deploy_server_model` is invoked, the server model is saved into the client's `FlexModel` under both `"model"` (local trainable copy) and `"server_model"` (frozen global reference):
```python
@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel) -> FlexModel:
    client_model = FlexModel()
    client_model["model"] = copy.deepcopy(server_flex_model["model"])
    client_model["server_model"] = copy.deepcopy(server_flex_model["model"])
    return client_model
```
Inside the client's `local_train` function, calculate the distance between parameters:
```python
prox_loss = sum(torch.sum((p - s_p.to(p.device)) ** 2)
                for p, s_p in zip(model.parameters(), server_model.parameters()))
loss = task_loss + (mu / 2.0) * prox_loss
```

### Pattern 2: Preserving Client State Across Federation Rounds (FedDyn & MOON)
FLEX allows storing arbitrary state in `client_flex_model`. When `@deploy_server_model` runs in subsequent rounds, existing keys on the client `FlexModel` that are not overwritten are preserved:
- **FedDyn**: Update and store client gradient memory $h_k^{t+1} = h_k^t - \alpha (\theta_k - \theta^t)$:
  ```python
  client_flex_model["feddyn_grad"] = updated_gradient_state
  ```
- **MOON**: Save the previous round's trained model as negative contrastive reference:
  ```python
  client_flex_model["prev_model"] = copy.deepcopy(model).cpu()
  ```

### Pattern 3: Step Tracking and Normalized Aggregation (FedNova)
When clients take variable numbers of local gradient steps:
1. Each client records its local iteration counter during training:
   ```python
   client_flex_model["fednova_iters"] = number_iterations
   ```
2. The aggregator retrieves the counts, normalizes them, and calls `weighted_fed_avg`:
   ```python
   iters = pool.clients.map(lambda cm, _: cm.get("fednova_iters", 1))
   weights = [c / sum(iters) for c in iters]
   pool.aggregators.map(weighted_fed_avg, ponderation=weights)
   ```

### Pattern 4: Hook-Based Representation Interception (MOON)
Instead of modifying the model class to output intermediate representations, register a forward pre-hook on the classification head (`fc`):
```python
captured = {}
def hook(module, inputs):
    captured["z"] = inputs[0]  # Penultimate feature representation

handle = model.fc.register_forward_pre_hook(hook)
try:
    model(x)
finally:
    handle.remove()
z = captured["z"]
```

---

## Running the Examples

All scripts support flexible command-line arguments:

### 1. FedAvg
```bash
python examples/01_fedavg.py --rounds 5 --clients 4 --epochs 3 --lr 0.05
```

### 2. FedProx
```bash
python examples/02_fedprox.py --rounds 5 --clients 4 --mu 0.01 --epochs 3
```

### 3. FedNova
```bash
python examples/03_fednova.py --rounds 5 --clients 4 --base-epochs 2
```

### 4. FedDyn
```bash
python examples/04_feddyn.py --rounds 5 --clients 4 --alpha 0.01 --epochs 3
```

### 5. MOON
```bash
python examples/05_moon.py --rounds 5 --clients 4 --mu 1.0 --tau 0.5 --epochs 3
```

### 6. Unified Comparison Showcase
To run all 5 baselines sequentially on the exact same data partition and display a comparative benchmark summary:
```bash
python examples/fl_framework_showcase.py --compare-all --rounds 5 --clients 4
```
Or run any individual baseline through the unified CLI:
```bash
python examples/fl_framework_showcase.py --method feddyn --rounds 5
```
