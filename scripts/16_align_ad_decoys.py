#!/usr/bin/env python
"""
16_align_ad_decoys.py — Per-target Cα Kabsch alignment of Chen et al. AD-decoy
poses (docked into DUD-E reference receptors) onto the DUD-Z receptor frame
used by the rest of the HDBind-3D pipeline.

Phase 3 of TIER3_DECOY_BIAS_PLAN.md, Branch B (coordinate-frame mismatch
between Chen's AD ligands and the DUD-Z receptors).

Pipeline per target:
  1. Parse Cα atoms from
       /home/maurbina/datasets/DUD-E_official/extracted/all/<target_lower>/receptor.pdb   (Chen frame)
       /home/maurbina/datasets/DUD-Z/<TARGET>/rec.crg.pdb                                  (DUD-Z frame)
  2. Pair residues by sequence identity using difflib.SequenceMatcher
     (handles residue-number offsets and chain-id mismatches).
  3. Solve R, t via Kabsch on paired Cα coordinates:  R @ src + t ≈ tgt.
  4. Apply (R, t) to all AD ligand conformers loaded from
       /home/maurbina/datasets/AD_decoys/102_AD_dataset/<target_lower>_AD.sdf
     and write transformed conformers to
       /home/maurbina/datasets/AD_decoys/transformed/<TARGET>/<target_lower>_AD_aligned.sdf
  5. Validate: post-alignment Cα RMSD < 2 Å, ligand-centroid post-shift within
     10 Å of the DUD-Z xtal-lig.pdb centroid.

Format note — the original spec called for `.mol2` outputs. RDKit has no mol2
writer in this venv and openbabel is not installed (and the project policy
forbids unjustified pip-installs). SDF preserves coordinates and topology
losslessly, matches the input format, and is consumable by the downstream
`iter_ad_poses` loader to be added in TIER3_DECOY_BIAS_PLAN Phase 3 step 1.

Usage:
    python scripts/16_align_ad_decoys.py                # smoke set (AA2AR ADRB2 EGFR)
    python scripts/16_align_ad_decoys.py --targets ALL  # all 41 AD-overlapping targets
    python scripts/16_align_ad_decoys.py --run-id smoke-3target
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import difflib
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from rdkit import Chem

# Make project imports work whether invoked from repo root or scripts/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import DUDEZ_ROOT, TARGETS_DEBUG, get_all_targets  # noqa: E402
from src.utils import setup_logging  # noqa: E402

log = logging.getLogger("16_align_ad_decoys")

# ── Paths ──────────────────────────────────────────────────────────────────
DUDE_RECEPTOR_ROOT = Path("/home/maurbina/datasets/DUD-E_official/extracted/all")
AD_SDF_ROOT = Path("/home/maurbina/datasets/AD_decoys/102_AD_dataset")
AD_TRANSFORMED_ROOT = Path("/home/maurbina/datasets/AD_decoys/transformed")
OUTPUT_ROOT = _REPO_ROOT / "outputs" / "16_align_ad_decoys"

# ── Validation thresholds ──────────────────────────────────────────────────
RMSD_THRESHOLD_A = 2.0       # post-Kabsch Cα RMSD must be < this
CENTROID_THRESHOLD_A = 10.0  # post-transform AD centroid → xtal-lig centroid

# ── Sequence-pairing parameters ────────────────────────────────────────────
MIN_BLOCK_SIZE = 5    # difflib matching block must span ≥ this many residues
MIN_PAIRED_CA = 30    # bail out if fewer than this many Cα atoms paired

# Three-letter → one-letter (canonical 20 + protonated histidine variants).
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Protonation-state variants → His
    "HIE": "H", "HID": "H", "HIP": "H",
    # Cysteine variants → Cys
    "CYX": "C", "CYM": "C",
    # Other rarely-seen aliases
    "MSE": "M",
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ PDB parsing                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class CAlpha:
    chain: str
    resseq: int
    icode: str
    resname: str
    coord: tuple[float, float, float]


_PDB_ATOM_RE = re.compile(r"^(ATOM  |HETATM)")


def parse_calphas(pdb_path: Path) -> list[CAlpha]:
    """Extract Cα atoms in file order. Skips alt-locs other than '' and 'A'."""
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")
    out: list[CAlpha] = []
    with open(pdb_path) as f:
        for line in f:
            if not _PDB_ATOM_RE.match(line):
                continue
            # Strict PDB column slicing.
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21] if len(line) > 21 else " "
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26] if len(line) > 26 else " "
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            out.append(CAlpha(
                chain=chain.strip() or "_",
                resseq=resseq,
                icode=icode.strip(),
                resname=resname,
                coord=(x, y, z),
            ))
    return out


def calphas_to_sequence(cas: list[CAlpha]) -> str:
    """One-letter sequence in file order; non-standard residues become 'X'."""
    return "".join(THREE_TO_ONE.get(ca.resname, "X") for ca in cas)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Sequence pairing                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def pair_calphas_by_sequence(
    src: list[CAlpha],
    tgt: list[CAlpha],
    min_block: int = MIN_BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """
    Pair Cα atoms between two receptors using one-letter sequence identity.

    Returns:
        src_xyz (N, 3), tgt_xyz (N, 3), index pairs [(src_i, tgt_j), ...]
    """
    src_seq = calphas_to_sequence(src)
    tgt_seq = calphas_to_sequence(tgt)
    matcher = difflib.SequenceMatcher(a=src_seq, b=tgt_seq, autojunk=False)
    pairs: list[tuple[int, int]] = []
    for block in matcher.get_matching_blocks():
        if block.size < min_block:
            continue
        for k in range(block.size):
            si, ti = block.a + k, block.b + k
            # Defensive: skip any 'X' (non-standard residue) pairings.
            if src_seq[si] == "X" or tgt_seq[ti] == "X":
                continue
            pairs.append((si, ti))
    if not pairs:
        return np.empty((0, 3)), np.empty((0, 3)), []
    src_xyz = np.array([src[si].coord for si, _ in pairs], dtype=np.float64)
    tgt_xyz = np.array([tgt[ti].coord for _, ti in pairs], dtype=np.float64)
    return src_xyz, tgt_xyz, pairs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Kabsch                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def kabsch(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (R, t) such that R @ src.T + t.reshape(3,1) ≈ tgt.T (column-vector
    convention). Equivalently, transformed = src @ R.T + t for row-vector
    points. Reflection-corrected via Kabsch's sign-flip.
    """
    assert src.shape == tgt.shape and src.shape[1] == 3, "src/tgt must be (N,3)"
    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    P = src - src_c
    Q = tgt - tgt_c
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = tgt_c - R @ src_c
    return R, t


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid transform to (N, 3) row-vector coords."""
    return coords @ R.T + t


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Reference-ligand centroid                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_xtal_lig_centroid(pdb_path: Path) -> np.ndarray:
    """Heavy-atom centroid of DUD-Z xtal-lig.pdb."""
    if not pdb_path.exists():
        raise FileNotFoundError(f"xtal-lig not found: {pdb_path}")
    coords: list[tuple[float, float, float]] = []
    with open(pdb_path) as f:
        for line in f:
            if not _PDB_ATOM_RE.match(line):
                continue
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if element == "H":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
    if not coords:
        raise ValueError(f"no heavy atoms parsed from {pdb_path}")
    return np.array(coords, dtype=np.float64).mean(axis=0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Per-target driver                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class TargetResult:
    target: str
    n_calphas_dude: int
    n_calphas_dudez: int
    n_paired: int
    pre_rmsd_a: float
    post_rmsd_a: float
    n_ad_ligands_in: int
    n_ad_ligands_out: int
    pre_centroid_offset_a: float
    post_centroid_offset_a: float
    rmsd_pass: bool
    centroid_pass: bool
    transform_path: str
    warnings: list[str]


def resolve_target_paths(target: str) -> dict[str, Path]:
    """Resolve all per-target inputs/outputs. Target is upper-case (e.g. AA2AR)."""
    upper = target.upper()
    lower = target.lower()
    return {
        "dude_receptor": DUDE_RECEPTOR_ROOT / lower / "receptor.pdb",
        "dudez_receptor": DUDEZ_ROOT / upper / "rec.crg.pdb",
        "xtal_lig": DUDEZ_ROOT / upper / "xtal-lig.pdb",
        "ad_sdf": AD_SDF_ROOT / f"{lower}_AD.sdf",
        "out_dir": AD_TRANSFORMED_ROOT / upper,
        "out_sdf": AD_TRANSFORMED_ROOT / upper / f"{lower}_AD_aligned.sdf",
    }


def transform_sdf(in_sdf: Path, out_sdf: Path, R: np.ndarray, t: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    """
    Stream ligands from in_sdf, apply (R, t) to all conformer atoms, write
    out_sdf. Returns (n_in, n_out, pre_centroid_mean, post_centroid_mean).
    """
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    pre_centroids: list[np.ndarray] = []
    post_centroids: list[np.ndarray] = []
    n_in = 0
    n_out = 0

    suppl = Chem.SDMolSupplier(str(in_sdf), removeHs=False, sanitize=False)
    writer = Chem.SDWriter(str(out_sdf))
    try:
        for mol in suppl:
            n_in += 1
            if mol is None:
                continue
            if mol.GetNumConformers() == 0:
                continue
            conf = mol.GetConformer(0)
            n_atoms = mol.GetNumAtoms()
            coords = np.empty((n_atoms, 3), dtype=np.float64)
            for i in range(n_atoms):
                p = conf.GetAtomPosition(i)
                coords[i] = (p.x, p.y, p.z)
            pre_centroids.append(coords.mean(axis=0))
            new_coords = apply_transform(coords, R, t)
            post_centroids.append(new_coords.mean(axis=0))
            for i in range(n_atoms):
                conf.SetAtomPosition(i, new_coords[i].tolist())
            writer.write(mol)
            n_out += 1
    finally:
        writer.close()

    if not pre_centroids:
        return n_in, 0, np.zeros(3), np.zeros(3)
    return (
        n_in,
        n_out,
        np.array(pre_centroids).mean(axis=0),
        np.array(post_centroids).mean(axis=0),
    )


def align_target(target: str) -> TargetResult:
    paths = resolve_target_paths(target)
    warnings: list[str] = []

    # 1) Receptor parsing.
    src_cas = parse_calphas(paths["dude_receptor"])
    tgt_cas = parse_calphas(paths["dudez_receptor"])
    log.info("[%s] Cα: DUD-E=%d, DUD-Z=%d", target, len(src_cas), len(tgt_cas))

    # 2) Pair by sequence.
    src_xyz, tgt_xyz, pairs = pair_calphas_by_sequence(src_cas, tgt_cas)
    n_paired = len(pairs)
    if n_paired < MIN_PAIRED_CA:
        raise RuntimeError(
            f"[{target}] only {n_paired} Cα paired (< {MIN_PAIRED_CA}); "
            f"sequences too divergent for confident alignment"
        )
    if n_paired < 0.5 * min(len(src_cas), len(tgt_cas)):
        warnings.append(
            f"low_pair_coverage={n_paired}/{min(len(src_cas), len(tgt_cas))}"
        )

    # 3) Kabsch.
    pre_rmsd = rmsd(src_xyz, tgt_xyz)
    R, t = kabsch(src_xyz, tgt_xyz)
    src_xform = apply_transform(src_xyz, R, t)
    post_rmsd = rmsd(src_xform, tgt_xyz)
    log.info("[%s] paired=%d  pre-RMSD=%.3f Å  post-RMSD=%.3f Å",
             target, n_paired, pre_rmsd, post_rmsd)

    # 4) Apply to AD ligands; capture pre/post centroids.
    n_in, n_out, pre_centroid, post_centroid = transform_sdf(
        paths["ad_sdf"], paths["out_sdf"], R, t
    )
    if n_out == 0:
        raise RuntimeError(f"[{target}] no AD ligands written from {paths['ad_sdf']}")
    if n_out < n_in:
        warnings.append(f"sdf_dropped={n_in - n_out}")

    # 5) Centroid validation against DUDE-Z xtal-lig.pdb.
    xtal_centroid = parse_xtal_lig_centroid(paths["xtal_lig"])
    pre_offset = float(np.linalg.norm(pre_centroid - xtal_centroid))
    post_offset = float(np.linalg.norm(post_centroid - xtal_centroid))
    log.info("[%s] AD-ligand-centroid offset: pre=%.2f Å  post=%.2f Å  "
             "(xtal_centroid=%s)", target, pre_offset, post_offset,
             np.round(xtal_centroid, 2).tolist())

    # 6) Persist transform alongside SDF for traceability.
    transform_npz = paths["out_dir"] / f"{target.lower()}_transform.npz"
    np.savez(transform_npz,
             R=R, t=t,
             pre_rmsd=pre_rmsd, post_rmsd=post_rmsd,
             n_paired=n_paired,
             xtal_centroid=xtal_centroid,
             pre_centroid=pre_centroid,
             post_centroid=post_centroid)

    return TargetResult(
        target=target,
        n_calphas_dude=len(src_cas),
        n_calphas_dudez=len(tgt_cas),
        n_paired=n_paired,
        pre_rmsd_a=pre_rmsd,
        post_rmsd_a=post_rmsd,
        n_ad_ligands_in=n_in,
        n_ad_ligands_out=n_out,
        pre_centroid_offset_a=pre_offset,
        post_centroid_offset_a=post_offset,
        rmsd_pass=post_rmsd < RMSD_THRESHOLD_A,
        centroid_pass=post_offset < CENTROID_THRESHOLD_A,
        transform_path=str(transform_npz),
        warnings=warnings,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Reporting                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def write_outputs(results: list[TargetResult], failures: dict[str, str], run_dir: Path,
                  run_id: str, targets_requested: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # per-target CSV
    per_target_csv = run_dir / "per_target.csv"
    with open(per_target_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "target", "n_calphas_dude", "n_calphas_dudez", "n_paired",
            "pre_rmsd_a", "post_rmsd_a",
            "n_ad_ligands_in", "n_ad_ligands_out",
            "pre_centroid_offset_a", "post_centroid_offset_a",
            "rmsd_pass", "centroid_pass", "transform_path", "warnings",
        ])
        for r in results:
            w.writerow([
                r.target, r.n_calphas_dude, r.n_calphas_dudez, r.n_paired,
                f"{r.pre_rmsd_a:.4f}", f"{r.post_rmsd_a:.4f}",
                r.n_ad_ligands_in, r.n_ad_ligands_out,
                f"{r.pre_centroid_offset_a:.4f}", f"{r.post_centroid_offset_a:.4f}",
                r.rmsd_pass, r.centroid_pass, r.transform_path,
                ";".join(r.warnings),
            ])

    # manifest.json
    manifest = {
        "run_id": run_id,
        "git_sha": _git_short_sha(),
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "script": "scripts/16_align_ad_decoys.py",
        "phase": "TIER3_DECOY_BIAS_PLAN Phase 3 — Branch B alignment",
        "thresholds": {
            "rmsd_a": RMSD_THRESHOLD_A,
            "centroid_a": CENTROID_THRESHOLD_A,
            "min_block_size": MIN_BLOCK_SIZE,
            "min_paired_ca": MIN_PAIRED_CA,
        },
        "input_paths": {
            "dude_receptor_root": str(DUDE_RECEPTOR_ROOT),
            "ad_sdf_root": str(AD_SDF_ROOT),
            "dudez_root": str(DUDEZ_ROOT),
        },
        "output_paths": {
            "transformed_sdf_root": str(AD_TRANSFORMED_ROOT),
            "report_dir": str(run_dir),
        },
        "output_format": "SDF (deviation from spec — see REPORT.md §Format note)",
        "targets_requested": targets_requested,
        "targets_succeeded": [r.target for r in results],
        "targets_failed": failures,
        "results": [asdict(r) for r in results],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # REPORT.md
    n_ok_rmsd = sum(r.rmsd_pass for r in results)
    n_ok_centroid = sum(r.centroid_pass for r in results)
    n_total = len(results)

    lines: list[str] = []
    lines.append(f"# Align AD decoys onto DUD-Z frame — `{run_id}`")
    lines.append("")
    lines.append(f"- **Script:** `scripts/16_align_ad_decoys.py`")
    lines.append(f"- **Git SHA:** `{manifest['git_sha']}`")
    lines.append(f"- **Timestamp:** {manifest['timestamp_utc']}")
    lines.append(f"- **Phase:** TIER3_DECOY_BIAS_PLAN Phase 3 — Branch B "
                 "(coordinate-frame mismatch)")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Per-target rigid-body Kabsch alignment of Chen et al.'s DUD-E "
                 "receptor (in whose frame the AD ligands are docked) onto the "
                 "DUD-Z `rec.crg.pdb` receptor used by the rest of the HDBind-3D "
                 "pipeline. Cα atoms paired by `difflib.SequenceMatcher` on "
                 "one-letter residue sequences (`min_block_size="
                 f"{MIN_BLOCK_SIZE}`), so residue-numbering offsets and chain-id "
                 "differences do not require manual curation. Resulting (R, t) "
                 "applied to every AD-ligand conformer; transformed conformers "
                 "written as one SDF per target.")
    lines.append("")
    lines.append("## Format note")
    lines.append("")
    lines.append("The phase spec calls for `.mol2` outputs. RDKit in this venv "
                 "has no mol2 writer and openbabel is not installed; the project "
                 "policy forbids unjustified `pip install`. SDF preserves "
                 "coordinates and topology losslessly, matches the input format, "
                 "and the downstream `iter_ad_poses` loader can consume it via "
                 "`Chem.SDMolSupplier` (Phase 3 step 1 of the bias plan).")
    lines.append("")
    lines.append("## Validation thresholds")
    lines.append("")
    lines.append(f"- Post-alignment Cα RMSD < **{RMSD_THRESHOLD_A:.1f} Å**")
    lines.append(f"- Post-transform AD ligand centroid within "
                 f"**{CENTROID_THRESHOLD_A:.1f} Å** of DUD-Z xtal-lig centroid")
    lines.append("")
    lines.append("## Gate readout")
    lines.append("")
    lines.append(f"- Targets attempted: **{len(targets_requested)}**")
    lines.append(f"- Targets succeeded: **{n_total}**  "
                 f"(failed: {len(failures)})")
    lines.append(f"- Cα RMSD gate: **{n_ok_rmsd}/{n_total}** pass")
    lines.append(f"- Ligand-centroid gate: **{n_ok_centroid}/{n_total}** pass")
    lines.append("")

    lines.append("## Per-target diagnostics")
    lines.append("")
    lines.append("| Target | nCα DUD-E | nCα DUD-Z | nPaired | preRMSD Å | postRMSD Å | "
                 "AD lig in/out | pre-centroidΔ Å | post-centroidΔ Å | "
                 "RMSD✓ | Cent✓ | Warnings |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---|")
    for r in sorted(results, key=lambda x: x.target):
        lines.append(
            f"| {r.target} | {r.n_calphas_dude} | {r.n_calphas_dudez} | "
            f"{r.n_paired} | {r.pre_rmsd_a:.2f} | {r.post_rmsd_a:.2f} | "
            f"{r.n_ad_ligands_in}/{r.n_ad_ligands_out} | "
            f"{r.pre_centroid_offset_a:.2f} | {r.post_centroid_offset_a:.2f} | "
            f"{'✓' if r.rmsd_pass else '✗'} | "
            f"{'✓' if r.centroid_pass else '✗'} | "
            f"{', '.join(r.warnings) if r.warnings else '—'} |"
        )
    lines.append("")

    if failures:
        lines.append("## Failures")
        lines.append("")
        for tgt, msg in failures.items():
            lines.append(f"- **{tgt}** — {msg}")
        lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Transformed ligand SDFs: `{AD_TRANSFORMED_ROOT}/<TARGET>/<target>_AD_aligned.sdf`")
    lines.append(f"- Per-target transform matrices: `<target>_transform.npz` "
                 "(R, t, pre/post RMSD, centroids)")
    lines.append(f"- This run: `{run_dir}` (manifest.json, per_target.csv, REPORT.md)")
    lines.append("")

    (run_dir / "REPORT.md").write_text("\n".join(lines))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CLI                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def resolve_targets(arg: list[str]) -> list[str]:
    if not arg:
        return list(TARGETS_DEBUG)
    if len(arg) == 1 and arg[0].upper() == "ALL":
        return [t for t in get_all_targets()]
    return [t.upper() for t in arg]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Target names (default: smoke set AA2AR ADRB2 EGFR; "
                             "use ALL for every DUD-Z target)")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier (default: YYYYMMDD-HHMMSS-<git-sha>)")
    args = parser.parse_args()

    setup_logging()
    targets = resolve_targets(args.targets)
    run_id = args.run_id or f"{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{_git_short_sha()}"
    run_dir = OUTPUT_ROOT / run_id

    log.info("Run ID: %s  |  Targets: %s", run_id, ", ".join(targets))
    log.info("Output: %s", run_dir)

    results: list[TargetResult] = []
    failures: dict[str, str] = {}
    for tgt in targets:
        try:
            results.append(align_target(tgt))
        except Exception as e:
            log.exception("[%s] alignment failed", tgt)
            failures[tgt] = f"{type(e).__name__}: {e}"

    write_outputs(results, failures, run_dir, run_id, targets)

    # Final stdout summary.
    log.info("─" * 60)
    log.info("Done. %d/%d succeeded.", len(results), len(targets))
    for r in results:
        flag = "OK " if (r.rmsd_pass and r.centroid_pass) else "WARN"
        log.info("  [%s] %s  postRMSD=%.2fÅ  centroidΔ=%.2fÅ  ligands=%d",
                 flag, r.target, r.post_rmsd_a, r.post_centroid_offset_a,
                 r.n_ad_ligands_out)
    log.info("REPORT: %s/REPORT.md", run_dir)

    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
