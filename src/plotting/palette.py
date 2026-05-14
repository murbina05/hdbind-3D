"""
Shared color palette and featurizer metadata for all publication plots.
Import from here — never define colors inline in plot scripts.
"""
from __future__ import annotations

import seaborn as sns

# ── Featurizer registry ────────────────────────────────────────────────────────
# Internal name (used in pkl paths, decoy_set CSVs, diagnostic CSVs)
FEATURIZER_INTERNAL: list[str] = [
    "plif",
    "plec",
    "grim",
    "pocket_featurizer",
    "distance_matrix",
    "distance_chemistry",
]

# Display labels for axes
FEATURIZER_LABELS: dict[str, str] = {
    "plif":               "PLIF",
    "plec":               "PLEC",
    "grim":               "GRIM",
    "pocket_featurizer":  "Pocket",
    "distance_matrix":    "Distance",
    "distance_chemistry": "DistChem",
}

# Comparison CSV uses display names → map to internal
DISPLAY_TO_INTERNAL: dict[str, str] = {v: k for k, v in FEATURIZER_LABELS.items()}

# ── Color palette ──────────────────────────────────────────────────────────────
# Colorblind-safe (Wong 2011 via seaborn "colorblind")
_RAW = sns.color_palette("colorblind", n_colors=len(FEATURIZER_INTERNAL))
FEATURIZER_COLORS: dict[str, tuple] = {
    feat: _RAW[i] for i, feat in enumerate(FEATURIZER_INTERNAL)
}

# ── Feature classification ─────────────────────────────────────────────────────
POCKET_FEATURES: list[str] = [
    "pocket_hydrophobic", "pocket_polar", "pocket_positive",
    "pocket_negative", "pocket_aromatic", "pocket_size",
]
LIGAND_FEATURES: list[str] = [
    "MolWt", "MolLogP", "NumHBD", "NumHBA", "TPSA",
    "NumRotBonds", "NumAromaticRings", "NumHeavyAtoms", "FractionCSP3",
]

FEATURE_COLORS: dict[str, str] = {
    "pocket": "#b0b0b0",   # light gray for pocket (constant) features
    "ligand": "#4878d0",   # steel blue for ligand descriptors
}

# ── Decoy set display names ────────────────────────────────────────────────────
DECOY_SET_LABELS: dict[str, str] = {
    "DUDE-Z default": "DUDE-Z default",
    "Extrema":        "Extrema",
    "Goldilocks":     "Goldilocks",
}

# ── matplotlib rcParams helpers ────────────────────────────────────────────────
def apply_publication_style() -> None:
    """Apply consistent publication-ready rcParams."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size":        10,
        "axes.titlesize":   12,
        "axes.labelsize":   10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.spines.top":  False,
        "axes.spines.right": False,
    })
