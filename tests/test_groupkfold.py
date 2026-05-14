"""Regression test for src.utils.load_canonical_groupkfold.

Phase 0 of TIER3_DECOY_BIAS_PLAN.md lifts the canonical split logic (formerly
inlined at scripts/07_train_egnn.py:445-462) into src/utils.py so eval-time
replay (10/11/12) cannot drift from training-time semantics. The prior bug
this prevents: an off-by-one fold-split inflated Goldilocks PR-AUC from 0.062
to 0.192. This test guards against re-introducing that drift.

Asserts the function, when given the m2-stratified manifest, replays the
fold composition recorded in that run's manifest.json (same val_targets and
n_train/n_val per fold).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from config import PROJECT_ROOT
from src.utils import load_canonical_groupkfold


M2_STRATIFIED_DIR = PROJECT_ROOT / "outputs" / "07_train_egnn" / "m2-stratified"


@pytest.fixture(scope="module")
def m2_stratified_manifest() -> dict:
    manifest_path = M2_STRATIFIED_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"m2-stratified manifest not found at {manifest_path}")
    return json.loads(manifest_path.read_text())


@pytest.fixture(scope="module")
def filtered_index(m2_stratified_manifest: dict) -> pd.DataFrame:
    dataset_dir = Path(m2_stratified_manifest["dataset_dir"])
    if not (dataset_dir / "index.csv").exists():
        pytest.skip(f"index.csv not found at {dataset_dir}")
    full = pd.read_csv(dataset_dir / "index.csv", dtype={"key": str})
    return (
        full[full["target"].isin(set(m2_stratified_manifest["targets"]))]
        .reset_index(drop=True)
    )


def test_load_canonical_groupkfold_matches_recorded_folds(
    m2_stratified_manifest: dict, filtered_index: pd.DataFrame,
) -> None:
    splits = load_canonical_groupkfold(M2_STRATIFIED_DIR / "manifest.json")
    assert len(splits) == m2_stratified_manifest["n_folds"]

    targets_arr = filtered_index["target"].values
    recorded_folds = m2_stratified_manifest["folds"]

    for fold_idx, ((train_i, val_i), recorded) in enumerate(zip(splits, recorded_folds)):
        assert len(train_i) == recorded["n_train"], (
            f"fold {fold_idx} n_train mismatch: "
            f"got {len(train_i)}, recorded {recorded['n_train']}"
        )
        assert len(val_i) == recorded["n_val"], (
            f"fold {fold_idx} n_val mismatch: "
            f"got {len(val_i)}, recorded {recorded['n_val']}"
        )

        observed_val_targets = sorted(set(targets_arr[val_i]))
        recorded_val_targets = sorted(recorded["val_targets"])
        assert observed_val_targets == recorded_val_targets, (
            f"fold {fold_idx} val_targets mismatch"
        )

        observed_train_targets = sorted(set(targets_arr[train_i]))
        recorded_train_targets = sorted(recorded["train_targets"])
        assert observed_train_targets == recorded_train_targets, (
            f"fold {fold_idx} train_targets mismatch"
        )


def test_load_canonical_groupkfold_train_pos_matches(
    m2_stratified_manifest: dict, filtered_index: pd.DataFrame,
) -> None:
    """Per-fold positive/negative counts in the train split must match the
    manifest record. This catches label-array misordering as well as wrong
    splits."""
    splits = load_canonical_groupkfold(M2_STRATIFIED_DIR / "manifest.json")
    labels_arr = filtered_index["label"].values.astype(int)

    for fold_idx, ((train_i, _val_i), recorded) in enumerate(
        zip(splits, m2_stratified_manifest["folds"])
    ):
        n_train_pos = int(labels_arr[train_i].sum())
        n_train_neg = int(len(train_i) - n_train_pos)
        assert n_train_pos == recorded["n_train_pos"], (
            f"fold {fold_idx} n_train_pos mismatch: "
            f"got {n_train_pos}, recorded {recorded['n_train_pos']}"
        )
        assert n_train_neg == recorded["n_train_neg"], (
            f"fold {fold_idx} n_train_neg mismatch: "
            f"got {n_train_neg}, recorded {recorded['n_train_neg']}"
        )
