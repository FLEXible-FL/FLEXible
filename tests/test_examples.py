"""Tests for deliverable FLEX examples."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


def _load_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "examples" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], str(script_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fedavg_mod():
    return _load_module("01_fedavg.py")


@pytest.fixture(scope="module")
def fedprox_mod():
    return _load_module("02_fedprox.py")


@pytest.fixture(scope="module")
def fednova_mod():
    return _load_module("03_fednova.py")


@pytest.fixture(scope="module")
def feddyn_mod():
    return _load_module("04_feddyn.py")


@pytest.fixture(scope="module")
def moon_mod():
    return _load_module("05_moon.py")


@pytest.fixture(scope="module")
def showcase_mod():
    return _load_module("fl_framework_showcase.py")


def test_fedavg_example_runs_cleanly(fedavg_mod):
    acc = fedavg_mod.run_fedavg(rounds=1, clients=2, epochs=1, batch_size=8)
    assert 0.0 <= acc <= 1.0


def test_fedprox_example_runs_cleanly(fedprox_mod):
    acc = fedprox_mod.run_fedprox(rounds=1, clients=2, mu=0.01, epochs=1, batch_size=8)
    assert 0.0 <= acc <= 1.0


def test_fednova_example_runs_cleanly(fednova_mod):
    acc = fednova_mod.run_fednova(rounds=1, clients=2, base_epochs=1, batch_size=8)
    assert 0.0 <= acc <= 1.0


def test_feddyn_example_runs_cleanly_with_state_update(feddyn_mod):
    # Run 2 rounds to verify client gradient memory is created in round 1 and reused in round 2
    acc = feddyn_mod.run_feddyn(rounds=2, clients=2, alpha=0.01, epochs=1, batch_size=8)
    assert 0.0 <= acc <= 1.0


def test_moon_example_runs_cleanly_with_contrastive_loss(moon_mod):
    # Run 2 rounds so prev_model is active in round 2
    acc = moon_mod.run_moon(rounds=2, clients=2, mu=1.0, tau=0.5, epochs=1, batch_size=8)
    assert 0.0 <= acc <= 1.0


@pytest.mark.parametrize("method", ["fedavg", "fedprox", "fednova", "feddyn", "moon"])
def test_showcase_simulation_runner(showcase_mod, method):
    loss, acc = showcase_mod.run_simulation(
        method=method, rounds=1, clients=2, epochs=1, batch_size=8, quiet=True
    )
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0
