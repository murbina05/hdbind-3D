# config.py
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path("/home/maurbina/hdbind-3D")
DUDEZ_ROOT    = Path("/home/maurbina/datasets/DUD-Z")
DATA_DIR      = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR   = PROJECT_ROOT / "results"
VENV_PYTHON   = Path("/home/maurbina/.venvs/dc_featurizers/bin/python")

DOCKING_DIR   = DUDEZ_ROOT          # targets live directly under root
MOL2_SUBDIR   = "DUDE_Z"           # subdirectory holding docked pose mol2 files
MOL2_PREFIX   = "dudez_1pt0LD"     # 1.0 Å ligand-diversity cutoff (default)

MOL2_SUBDIRS = {
    "DUDE_Z":     ("DUDE_Z",     "dudez_1pt0LD"),
    "Extrema":    ("Extrema",    "extrema_1pt0LD"),
    "Goldilocks": ("Goldilocks", "goldilocks_1pt0LD"),
}

# ── Hardware ───────────────────────────────────────────────────────────────
N_WORKERS         = min(32, os.cpu_count() or 1)   # Threadripper 9970X
NESTED_WORKERS    = 4                               # Per-target inner pool
OMP_THREAD_LIMIT  = 4                              # Prevent thread explosion

# GPU — RTX PRO 6000 Blackwell (96 GB VRAM)
import torch as _torch
GPU_AVAILABLE     = _torch.cuda.is_available()
GPU_DEVICE        = "cuda:0" if GPU_AVAILABLE else "cpu"
GPU_BATCH_SIZE    = 4096    # Ligands per GPU batch for distance/projection ops
GPU_VRAM_LIMIT_GB = 80.0    # Leave headroom; allocate up to this



# XGBoost GPU config (used in training.py)
XGBOOST_PARAMS = {
    "tree_method": "hist",
    "device": "cuda" if GPU_AVAILABLE else "cpu",
    "max_bin": 256,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 30,  # ~30:1 decoy:active ratio
    "random_state": 42,
}

# cuML availability (RAPIDS — optional, falls back to sklearn)
try:
    import cuml as _cuml
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# ── Featurization ──────────────────────────────────────────────────────────
POCKET_CUTOFF_ANG = 6.0        # Å — pocket residue definition
HYDROPHOBIC_CUTOFF = 4.5       # Å — hydrophobic contact threshold
PLEC_DEPTH_LIGAND = 2
PLEC_DEPTH_PROTEIN = 4
PLEC_SIZE = 4096
PLEC_CONTACT_CUTOFF = 4.5   # Å — ligand–protein atom pair close-contact threshold
HD_DIM = 10000
RANDOM_SEED = 42

PROLIF_INTERACTIONS = [
    "Hydrophobic", "HBDonor", "HBAcceptor", "Cationic", "Anionic",
    "CationPi", "PiCation", "PiStacking", "EdgeToFace", "FaceToFace",
    "XBAcceptor", "XBDonor", "MetalDonor", "MetalAcceptor",
]

RESIDUE_PROPERTIES = {
    # [hydrophobic, polar, positive, negative, aromatic]
    "ALA": [1, 0, 0, 0, 0], "ARG": [0, 1, 1, 0, 0], "ASN": [0, 1, 0, 0, 0],
    "ASP": [0, 1, 0, 1, 0], "CYS": [1, 0, 0, 0, 0], "GLN": [0, 1, 0, 0, 0],
    "GLU": [0, 1, 0, 1, 0], "GLY": [0, 0, 0, 0, 0], "HIS": [0, 1, 1, 0, 1],
    "ILE": [1, 0, 0, 0, 0], "LEU": [1, 0, 0, 0, 0], "LYS": [0, 1, 1, 0, 0],
    "MET": [1, 0, 0, 0, 0], "PHE": [1, 0, 0, 0, 1], "PRO": [1, 0, 0, 0, 0],
    "SER": [0, 1, 0, 0, 0], "THR": [0, 1, 0, 0, 0], "TRP": [0, 1, 0, 0, 1],
    "TYR": [0, 1, 0, 0, 1], "VAL": [1, 0, 0, 0, 0], "UNK": [0, 0, 0, 0, 0],
}

# ── Evaluation ─────────────────────────────────────────────────────────────
CV_FOLDS = 5
CV_SEED  = 42
BEDROC_ALPHA = 20.0
EF_FRACTIONS = [0.01, 0.05]
THRESHOLD_SCAN = [round(t, 2) for t in [i * 0.05 for i in range(2, 19)]]  # 0.10–0.90

# ── Targets ────────────────────────────────────────────────────────────────
_NON_TARGETS = {"dockfiles"}

def get_all_targets() -> list[str]:
    """Return sorted list of all 43 DUDE-Z target names."""
    return sorted([d.name for d in DOCKING_DIR.iterdir()
                   if d.is_dir() and d.name not in _NON_TARGETS])

import socket
HOSTNAME = socket.gethostname()
DEBUG_MODE = HOSTNAME == "seelab-langley"  # auto-detects which machine you're on

TARGETS_DEBUG = ["AA2AR", "ADRB2", "EGFR"]  # small, medium, large pocket sizes
TARGETS_ALL   = get_all_targets()
TARGETS        = TARGETS_DEBUG if DEBUG_MODE else TARGETS_ALL