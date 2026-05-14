#!/usr/bin/env python
"""Extract FLOWR.ROOT v2.1 pretrained embeddings — Phase 2 (SDF) + Phase 3 (DUDE-Z mol2).

Per-ligand loop. For each ligand, runs `predict_affinity_batch` at noise_scale=0.0,
captures pre- and post-MLP pooled features via 6 forward hooks on the LigandDecoder
pool modules plus a 7th validation hook on `pic50_head[0]`, and writes a per-pose
`.npz`.

Two input modes (mutually exclusive):
  --ligand_file <sdf>      Phase 2 mode. One SDF, no labels. Used for BACE smoke.
  --target <name>          Phase 3 mode. Iterates DUDE-Z primary poses via
                           src.data_loading.iter_poses(target, 'ligand'|'decoy').
                           PDB resolves to DOCKING_DIR/<target>/rec.crg.pdb.
                           Each pose's .npz includes `label` (1=active, 0=decoy)
                           and `kind` ('ligand' or 'decoy').

Validation built-in per ligand:
  1. Fire-count assertion (each hook fires exactly twice — pass-1 + pass-2).
  2. Algebraic check: `concat(z_lig_post, z_pocket_post, z_int_post) == combined`.
  3. Semantic check: manually running the four affinity head MLPs on the captured
     combined features reproduces the model's returned affinity scalars.

Forked from /home/maurbina/flowr_root/scripts/export_embeddings.py — we keep the
ligand-prep helpers and the SimpleNamespace args pattern; we replace the post-only
EmbeddingCapture with a fuller HookCapture.

MDAnalysis note (project lesson): we do NOT touch `mda.Universe` here. FLOWR.ROOT's
own pocket cutter (via ProDy + biopython, invoked through `load_data_from_pdb`)
handles all protein I/O. If you add MDAnalysis-based logic later, build a fresh
`mda.Universe` per residue per CLAUDE.md — sharing universes across residues
silently corrupts bond topology in this codebase.

Hard constraints (brief §5):
  - No edits to FLOWR.ROOT source. All hook code lives in this script.
  - dc_featurizers venv only.
  - Serial loop; no GPU multiprocessing.
  - noise_scale=0.0; t flow time is internally pinned to ~1.0 by
    predict_affinity_batch (see notes/hook_locations.md §4).
  - DUDE-Z mol2 names come from the `########## Name:` header, NEVER the
    `Long Name:` line which is universally `NO_LONG_NAME` (see
    notes/phase3_mol2_audit.md). Delegate parsing to src.data_loading.iter_poses.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import re
import sys
import tempfile
import threading
import time
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_FLOWR = Path("/home/maurbina/flowr_root")
_HDBIND = Path("/home/maurbina/hdbind-3D")
for _p in (_HDBIND, _FLOWR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# Project helpers — only imported when --target mode is used.
# (Importing at module top is fine; config.py loads quickly.)
from src.data_loading import iter_poses  # noqa: E402
from config import DOCKING_DIR  # noqa: E402

from flowr.data.dataset import GeometricDataset
from flowr.gen.utils import get_dataloader, load_data_from_pdb, load_util
from flowr.predict.predict import predict_affinity_batch
from flowr.scriptutil import load_model
from flowr.util.device import get_device
from flowr.util.pocket import PocketComplexBatch


DEFAULT_CKPT = _FLOWR / "checkpoints" / "flowr_root_v2.1.ckpt"


# ---------------------------------------------------------------------------
# Ligand prep — verbatim from export_embeddings.py
# ---------------------------------------------------------------------------

def _fix_valence(mol):
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
        mol_noH = Chem.RemoveAllHs(mol, sanitize=False)
        mol_noH.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol_noH)
        return Chem.AddHs(mol_noH, addCoords=True)
    except Exception:
        pass
    try:
        mol2 = Chem.RWMol(mol)
        mol2.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol2)
        return mol2.GetMol()
    except Exception:
        return None


def _write_mol_to_tmp_sdf(mol: Chem.Mol) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    w = Chem.SDWriter(tmp.name)
    w.write(mol)
    w.close()
    return Path(tmp.name)


def enumerate_target_poses_one_sdf(target: str) -> tuple[Path, list[dict]]:
    """Phase 4 Step 3 helper — write ALL parsed poses for a target into a single
    multi-mol SDF, so FLOWR's `--multiple_ligands` codepath can be reused for
    batched extraction.

    Returns (sdf_path, records). `records[i]` corresponds to ligand_idx=i in the
    SDF; each record has {idx, name, label, kind, recovered=False}.
    The caller is responsible for unlinking the SDF when done.
    """
    tmp_sdf = Path(tempfile.NamedTemporaryFile(suffix=".sdf", delete=False).name)
    writer = Chem.SDWriter(str(tmp_sdf))
    records: list[dict] = []
    sdf_idx = 0
    try:
        for kind, label in [("ligand", 1), ("decoy", 0)]:
            for mol_id, mol in iter_poses(target, kind):
                mol.SetProp("_Name", mol_id)
                writer.write(mol)
                records.append({
                    "idx": sdf_idx,
                    "name": mol_id,
                    "label": label,
                    "kind": kind,
                    "recovered": False,
                })
                sdf_idx += 1
    finally:
        writer.close()
    return tmp_sdf, records


def enumerate_target_poses(target: str) -> list[dict]:
    """Phase 3 input mode — enumerate DUDE-Z primary poses for a target.

    Uses the project's `iter_poses(target, kind)` which:
      - Reads the mol2 `########## Name:` header (NOT `Long Name:` which is
        always `NO_LONG_NAME` in DUDE-Z; see notes/phase3_mol2_audit.md).
      - Sanitizes via Chem.SanitizeMol with silent drops for failures.

    Writes each successfully-parsed mol to a tmp single-mol SDF (preserves the
    3D conformer) and returns a list of records:

        {idx, name, tmp_path, label, kind, recovered}

    where:
      idx       — global enumeration index (actives first, then decoys)
      name      — mol2 Name (CHEMBL... for actives, ZINC... for decoys)
      tmp_path  — Path to the single-mol tmp SDF (caller must unlink after use)
      label     — 1 for actives ('ligand' kind), 0 for decoys ('decoy' kind)
      kind      — 'ligand' or 'decoy' (matches iter_poses() arg)
      recovered — False; mol2 parsing has no fix-valence rescue path here.
                  Silent drops are surfaced via len(records) < grep MOLECULE count.
    """
    records: list[dict] = []
    idx = 0
    for kind, label in [("ligand", 1), ("decoy", 0)]:
        for mol_id, mol in iter_poses(target, kind):
            tmp = _write_mol_to_tmp_sdf(mol)
            records.append({
                "idx": idx,
                "name": mol_id,
                "tmp_path": tmp,
                "label": label,
                "kind": kind,
                "recovered": False,
            })
            idx += 1
    return records


def split_sdf_to_single_ligand_files(sdf_path: Path) -> list[tuple[int, str, Path, bool]]:
    """Read a multi-ligand SDF and split into per-ligand tmp SDFs.

    Returns a list of (index, name, tmp_path, recovered_via_fix_valence).
    Drops mols that cannot be sanitized even after the fix-valence rescue.
    """
    out: list[tuple[int, str, Path, bool]] = []
    suppl_strict = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    suppl_raw = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)

    strict_mols = list(suppl_strict)
    raw_mols = list(suppl_raw)
    assert len(strict_mols) == len(raw_mols), "SDF mol-count mismatch between strict/raw"

    for idx, (m_strict, m_raw) in enumerate(zip(strict_mols, raw_mols)):
        mol = m_strict
        recovered = False
        if mol is None and m_raw is not None:
            mol = _fix_valence(m_raw)
            recovered = mol is not None
        if mol is None:
            continue  # caller will see the gap via failures list
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"lig_{idx:04d}"
        tmp = _write_mol_to_tmp_sdf(mol)
        out.append((idx, name, tmp, recovered))
    return out


# ---------------------------------------------------------------------------
# Hook capture — 6 production hooks (3 pre + 3 post) + 1 validation hook
# ---------------------------------------------------------------------------

# Expected fires per hook per predict_affinity_batch call:
#   _predict_affinity does TWO forward passes (self-conditioning prior + clean pose).
#   Both passes traverse the same path → each hook fires exactly twice.
#   See notes/hook_locations.md §4 for derivation.
EXPECTED_FIRES_PER_HOOK = 2

HOOK_NAMES = [
    "f_lig_pre",
    "f_pocket_pre",
    "eij_pooled_pre",
    "z_lig_post",
    "z_pocket_post",
    "z_int_post",
    "combined_at_pic50_head",
]


class PocketEncoderCache:
    """Monkey-patches `model.gen.pocket_enc.forward` to return a cached output.

    Phase 4 finding (2026-05-11): FLOWR.ROOT's `process_complex` re-cuts the
    pocket around each ligand pose's heavy atoms with a fixed cutoff. For DUDE-Z
    AA2AR, observed pocket sizes across the first 50 poses ranged 199→314 atoms.
    So even though `rec.crg.pdb` is shared per target, the per-pose pocket cut
    is NOT bit-identical → full cross-pose caching would change `pocket_enc`
    inputs and produce drift beyond Phase 9's BF16 tolerance.

    This class is therefore disabled by default. The shape-signature assertion
    will fire-loud if it's installed and inputs differ — preventing silent
    correctness regressions. Kept in tree for two valid uses:

      1. Within-pose caching across the two forward passes of `_predict_affinity`
         (pass-1 and pass-2 share the same pocket_data → safe to cache).
         Estimated savings: ~9 ms/pose (1.06× total — below the brief's
         <1.5× threshold, so skipped from production).
      2. A future refactor where pocket cuts are fixed per target (e.g., cut
         once around the crystal ligand `xtal-lig.pdb`). Would change feature
         semantics → out of scope here.

    Lifetime: .install() once before processing a target; .reset() between
    targets; .uninstall() once at the end.
    """

    def __init__(self, pocket_enc: torch.nn.Module):
        self.pocket_enc = pocket_enc
        self._original_forward = None
        self.cache_value = None
        self.input_signature: tuple | None = None
        self.n_hits = 0
        self.n_misses = 0

    @staticmethod
    def _shape_signature(args) -> tuple:
        sig: list = []
        for a in args[:6]:
            if torch.is_tensor(a):
                sig.append(("T", tuple(a.shape), str(a.dtype)))
            elif a is None:
                sig.append(("N",))
            else:
                sig.append(("?",))
        return tuple(sig)

    def _cached_forward(self, *args, **kwargs):
        sig = self._shape_signature(args)
        if self.cache_value is None:
            self.cache_value = self._original_forward(*args, **kwargs)
            self.input_signature = sig
            self.n_misses += 1
        else:
            if sig != self.input_signature:
                raise AssertionError(
                    "PocketEncoderCache input-shape signature changed without "
                    "reset() — pocket cut likely differs across poses. "
                    f"cached={self.input_signature} got={sig}. "
                    "Call .reset() when switching targets."
                )
            self.n_hits += 1
        return self.cache_value

    def install(self) -> None:
        if self._original_forward is None:
            self._original_forward = self.pocket_enc.forward
        self.pocket_enc.forward = self._cached_forward

    def uninstall(self) -> None:
        if self._original_forward is not None:
            self.pocket_enc.forward = self._original_forward
            self._original_forward = None

    def reset(self) -> None:
        """Invalidate the cached pocket-encoder output (call between targets)."""
        self.cache_value = None
        self.input_signature = None


class HookCapture:
    """Manages forward hooks on the LigandDecoder pool modules + pic50_head[0].

    Each hook appends to a list rather than overwriting, so pass-1 and pass-2
    captures are both available for inspection. `pass2()` returns the second
    capture per hook (the clean-pose / affinity-producing pass). `assert_fired()`
    validates the per-hook fire count against EXPECTED_FIRES_PER_HOOK and is
    intended to fail loudly if a future FLOWR.ROOT change breaks the
    two-pass-per-call assumption.
    """

    def __init__(self, decoder: torch.nn.Module):
        self.decoder = decoder
        self.captures: dict[str, list[torch.Tensor]] = {n: [] for n in HOOK_NAMES}
        self.fire_counts: dict[str, int] = defaultdict(int)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _record(self, name: str, tensor: torch.Tensor) -> None:
        self.captures[name].append(tensor.detach().cpu())
        self.fire_counts[name] += 1

    def _pre(self, name: str):
        def hook(module, args):
            t = args[0] if isinstance(args, (tuple, list)) and args else args
            self._record(name, t)
        return hook

    def _post(self, name: str):
        def hook(module, inputs, output):
            self._record(name, output)
        return hook

    def _head_input(self, name: str):
        def hook(module, inputs, output):
            t = inputs[0] if isinstance(inputs, (tuple, list)) and inputs else inputs
            self._record(name, t)
        return hook

    def register(self) -> None:
        dec = self.decoder
        self._handles = [
            dec.lig_pool.register_forward_pre_hook(self._pre("f_lig_pre")),
            dec.pocket_pool.register_forward_pre_hook(self._pre("f_pocket_pre")),
            dec.interaction_pool.register_forward_pre_hook(self._pre("eij_pooled_pre")),
            dec.lig_pool.register_forward_hook(self._post("z_lig_post")),
            dec.pocket_pool.register_forward_hook(self._post("z_pocket_post")),
            dec.interaction_pool.register_forward_hook(self._post("z_int_post")),
            dec.pic50_head[0].register_forward_hook(
                self._head_input("combined_at_pic50_head")
            ),
        ]

    def reset(self) -> None:
        for n in HOOK_NAMES:
            self.captures[n] = []
            self.fire_counts[n] = 0

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def assert_fired(self, expected: int = EXPECTED_FIRES_PER_HOOK) -> None:
        bad = {n: c for n, c in self.fire_counts.items() if c != expected}
        if bad:
            raise AssertionError(
                f"Hook fire-count assertion FAILED. Expected {expected} fires per hook "
                f"(one per pass-1/pass-2 in _predict_affinity). Got mismatches: {bad}. "
                "If FLOWR.ROOT's affinity-prediction path was changed upstream, "
                "review notes/hook_locations.md §4 and update EXPECTED_FIRES_PER_HOOK."
            )

    def pass2(self) -> dict[str, torch.Tensor]:
        """Return the pass-2 capture for each hook (clean-pose, affinity-producing)."""
        out = {}
        for n in HOOK_NAMES:
            caps = self.captures[n]
            if len(caps) < 2:
                raise AssertionError(
                    f"Hook {n!r} fired only {len(caps)} times; expected ≥2. "
                    "Fire-count assertion should have caught this earlier."
                )
            out[n] = caps[1]
        return out

    def pass1(self) -> dict[str, torch.Tensor]:
        return {n: self.captures[n][0] for n in HOOK_NAMES}


# ---------------------------------------------------------------------------
# Phase 5 all-targets orchestration: persistent worker pool, heartbeat,
# resume-on-existing-manifest, nested tqdm.
# ---------------------------------------------------------------------------

def _list_all_dudez_targets() -> list[str]:
    """All 43 DUDE-Z target subdirectories that have a rec.crg.pdb."""
    targets = []
    for d in sorted(DOCKING_DIR.iterdir()):
        if d.is_dir() and (d / "rec.crg.pdb").exists():
            targets.append(d.name)
    return targets


def _target_is_complete(target: str, features_root: Path) -> tuple[bool, dict | None]:
    """A target is 'complete' if features_root/<target>/manifest.json reports
    n_extracted + n_failed == n_readable.

    The right denominator is `n_readable` (post `iter_poses` RDKit-sanitize
    filter), NOT `n_raw_in_source` (post-grep MOLECULE count). iter_poses
    silently drops mols that fail RDKit sanitization; those drops aren't a
    pipeline failure to retry — they're an upstream-data limitation.
    """
    mp = features_root / target / "manifest.json"
    if not mp.exists():
        return False, None
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return False, None
    if m.get("n_extracted") is None:
        return False, m
    target_n = m.get("n_readable", m.get("n_raw_in_source", -1))
    if m["n_extracted"] + m.get("n_failed", 0) == target_n and m["n_extracted"] > 0:
        return True, m
    return False, m


class _HeartbeatState:
    """Atomic shared state for the heartbeat daemon thread (Phase 5 §4.5)."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.current_target: str | None = None
        self.poses_done = 0
        self.poses_total = 0
        self.targets_done = 0
        self.targets_total = 0
        self.stop = False

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "current_target": self.current_target,
                "poses_done": self.poses_done,
                "poses_total": self.poses_total,
                "targets_done": self.targets_done,
                "targets_total": self.targets_total,
            }


def _heartbeat_loop(state: _HeartbeatState, interval_s: int = 60) -> None:
    """Daemon thread: every interval, log GPU util/mem + current target progress.

    Uses `tqdm.write` (NOT print) so the nested progress bars render correctly.
    """
    import subprocess
    while True:
        # Sleep in small slices so .stop is honored promptly
        for _ in range(interval_s):
            if state.stop:
                return
            time.sleep(1)
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            util, mem_mb = [p.strip() for p in r.stdout.strip().split(",")]
            mem_gb = int(mem_mb) / 1024.0
            s = state.snapshot()
            ts = time.strftime("%H:%M:%S")
            line = (
                f"[hb {ts}] target {s['targets_done']+1}/{s['targets_total']} "
                f"= {s['current_target']}  "
                f"poses {s['poses_done']}/{s['poses_total']}  "
                f"gpu_util={util}%  gpu_mem={mem_gb:.1f}GB"
            )
            tqdm.write(line)
        except Exception as exc:
            tqdm.write(f"[hb] sample failed: {exc}")


# ---------------------------------------------------------------------------
# Args builder — mirrors export_embeddings.make_base_args
# ---------------------------------------------------------------------------

def make_base_args(ckpt_path: str, save_dir: str, num_workers: int, seed: int,
                   batch_cost: int, coord_noise_scale: float, pocket_cutoff: float):
    return SimpleNamespace(
        ckpt_path=ckpt_path, arch="pocket", pocket_type="holo",
        pocket_noise="fix", lora_finetuned=False,
        cut_pocket=True, pocket_cutoff=pocket_cutoff,
        protonate_pocket=False, max_pocket_size=1000, min_pocket_size=10,
        compute_interactions=False, compute_interaction_recovery=False,
        add_hs=False, add_hs_and_optimize=False, add_hs_and_optimize_gen_ligs=False,
        kekulize=False, use_pdbfixer=False, add_bonds_to_protein=False,
        add_hs_to_protein=False, pocket_coord_noise_std=0.0,
        scaffold_hopping=False, scaffold_elaboration=False,
        linker_inpainting=False, fragment_inpainting=False,
        fragment_growing=False, core_growing=False,
        substructure_inpainting=False, interaction_conditional=False,
        interaction_inpainting=False, fixed_interactions=False,
        anisotropic_prior=False, substructure=None, graph_inpainting=None,
        max_fragment_cuts=3,
        coord_noise_scale=coord_noise_scale, sample_mol_sizes=False,
        corrector_iters=0, rotation_alignment=False,
        permutation_alignment=False, use_sde_simulation=False,
        use_cosine_scheduler=False, integration_steps=100,
        cat_sampling_noise_level=1, ode_sampling_strategy="linear",
        categorical_strategy="uniform-sample", solver="euler",
        bucket_cost_scale="quadratic",
        ligand_time=None, pocket_time=None, interaction_time=None,
        separate_pocket_interpolation=False,
        separate_interaction_interpolation=False,
        gpus=1, num_workers=num_workers, seed=seed, batch_cost=batch_cost,
        mp_index=0, save_dir=save_dir, data_path=None, dataset=None,
        pdb_id=None, ligand_id=None, pdb_file=None, ligand_file=None,
        res_txt_file=None, multiple_ligands=False,
        lr=None, lr_schedule=None, cosine_decay_fraction=None,
    )


def get_dataset(system, transform, vocab, interpolant, args, hparams):
    systems = PocketComplexBatch(system if isinstance(system, list) else [system])
    return GeometricDataset(systems, data_cls=PocketComplexBatch, transform=transform)


# ---------------------------------------------------------------------------
# Per-ligand processing
# ---------------------------------------------------------------------------

def _affinity_props_from_mol(mol: Chem.Mol) -> dict[str, float]:
    """Read pic50/pkd/pki/pec50 props set by predict_affinity_batch."""
    out: dict[str, float] = {}
    for key in ("pic50", "pkd", "pki", "pec50"):
        if mol is not None and mol.HasProp(key):
            try:
                out[key] = float(mol.GetProp(key))
            except ValueError:
                out[key] = float("nan")
        else:
            out[key] = float("nan")
    return out


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)[:120] or "ligand"


def _run_heads_manually(decoder: torch.nn.Module, combined: torch.Tensor) -> dict[str, float]:
    """Run the 4 head MLPs on the captured combined_features and return scalars.

    Used as a semantic validation that the captured combined corresponds to the
    *affinity-producing* (pass-2) forward call. Returns one scalar per head
    matching what the model's TensorDict would emit for B=1.
    """
    device = next(decoder.parameters()).device
    combined = combined.to(device)
    with torch.no_grad():
        out = {
            "pic50": decoder.pic50_head(combined),
            "pkd": decoder.pkd_head(combined),
            "pki": decoder.pki_head(combined),
            "pec50": decoder.pec50_head(combined),
        }
    return {k: float(v.squeeze().item()) for k, v in out.items()}


def process_one_ligand(
    idx: int,
    name: str,
    sdf_path: Path,
    pdb_path: Path,
    args_template: SimpleNamespace,
    model,
    hparams: dict,
    vocab,
    vocab_charges,
    vocab_hybridization,
    vocab_aromatic,
    transform,
    interpolant,
    hooks: HookCapture,
    output_dir: Path,
    label: int | None = None,
    kind: str | None = None,
    head_tolerance: float = 1e-3,
    concat_tolerance: float = 1e-5,
    stage_log: dict | None = None,
    prebuilt_system=None,
) -> dict:
    """Process one ligand: forward pass + capture + validate + write .npz."""
    args = SimpleNamespace(**vars(args_template))
    args.pdb_id = f"lig_{idx:04d}"
    args.pdb_file = str(pdb_path)
    args.ligand_file = str(sdf_path)
    args.multiple_ligands = False

    t_start = time.perf_counter()
    if prebuilt_system is not None:
        system = prebuilt_system
    else:
        system = load_data_from_pdb(
            args,
            remove_hs=hparams["remove_hs"],
            remove_aromaticity=hparams["remove_aromaticity"],
        )
    if isinstance(system, list):
        system = system[:1]
    t_after_load = time.perf_counter()
    dataset = get_dataset(system, transform, vocab, interpolant, args, hparams)
    dataloader = get_dataloader(args, dataset, interpolant)
    batch = next(iter(dataloader))
    prior, posterior, _, _ = batch
    t_after_dataloader = time.perf_counter()

    hooks.reset()
    gen_ligs = predict_affinity_batch(
        args, model=model, prior=prior, posterior=posterior,
        noise_scale=args.coord_noise_scale, eps=1e-4, seed=args.seed,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # ensure GPU completes before stage timer
    t_after_predict = time.perf_counter()

    elapsed_ms = (t_after_predict - t_start) * 1000.0
    if stage_log is not None:
        stage_log.setdefault("load_data_ms", []).append(
            (t_after_load - t_start) * 1000.0
        )
        stage_log.setdefault("dataloader_ms", []).append(
            (t_after_dataloader - t_after_load) * 1000.0
        )
        stage_log.setdefault("predict_ms", []).append(
            (t_after_predict - t_after_dataloader) * 1000.0
        )

    # Validation #1 — fire count assertion (raises on failure)
    hooks.assert_fired(expected=EXPECTED_FIRES_PER_HOOK)

    # Snapshot fire_counts NOW, before _run_heads_manually below.
    # _run_heads_manually invokes decoder.pic50_head(...), which re-fires the
    # `combined_at_pic50_head` hook on `pic50_head[0]` and would inflate that
    # hook's recorded count past EXPECTED_FIRES_PER_HOOK for the manifest
    # aggregator. The per-ligand assertion above is the load-bearing check;
    # this snapshot is just for downstream reporting.
    fire_counts_snapshot = dict(hooks.fire_counts)

    pass2 = hooks.pass2()

    # Shape sanity
    expected_shapes = {
        "f_lig_pre": 1024,
        "f_pocket_pre": 512,
        "eij_pooled_pre": 128,
        "z_lig_post": 128,
        "z_pocket_post": 128,
        "z_int_post": 128,
        "combined_at_pic50_head": 384,
    }
    for k, expected_dim in expected_shapes.items():
        t = pass2[k]
        if t.dim() != 2 or t.shape[0] != 1 or t.shape[1] != expected_dim:
            raise AssertionError(
                f"Ligand {idx} ({name}): hook {k!r} unexpected shape "
                f"{tuple(t.shape)}, want (1, {expected_dim})"
            )
        if not torch.isfinite(t).all():
            raise AssertionError(
                f"Ligand {idx} ({name}): NaN/Inf in {k!r}"
            )

    # Validation #2 — algebraic check: concat(z_*) == combined
    concat = torch.cat(
        [pass2["z_lig_post"], pass2["z_pocket_post"], pass2["z_int_post"]],
        dim=-1,
    )
    combined = pass2["combined_at_pic50_head"]
    concat_diff = (concat - combined).abs().max().item()
    if concat_diff > concat_tolerance:
        raise AssertionError(
            f"Ligand {idx} ({name}): concat(z_*_post) != combined_at_pic50_head; "
            f"max |diff|={concat_diff:.3e} (tol={concat_tolerance})"
        )

    # Validation #3 — semantic: heads(combined) ≈ predict_affinity_batch's returned values
    model_affinity = _affinity_props_from_mol(gen_ligs[0])
    manual_affinity = _run_heads_manually(model.gen.ligand_dec, combined)
    head_diffs = {
        k: abs(model_affinity[k] - manual_affinity[k]) for k in ("pic50", "pkd", "pki", "pec50")
    }
    worst_head_diff = max(head_diffs.values())
    if worst_head_diff > head_tolerance:
        raise AssertionError(
            f"Ligand {idx} ({name}): manual head run does not match predict_affinity_batch. "
            f"diffs={head_diffs}; tol={head_tolerance}. "
            "Captured combined features may not correspond to pass 2."
        )

    # Write .npz
    t_before_write = time.perf_counter()
    feature_arrays = {
        k: pass2[k].squeeze(0).float().numpy().astype(np.float32) for k in HOOK_NAMES
    }
    label_tag = f"a{label}" if label is not None else "u"
    fname = output_dir / f"{idx:05d}_{label_tag}_{_sanitize_filename(name)}.npz"
    extra: dict = {}
    if label is not None:
        extra["label"] = np.int32(label)
    if kind is not None:
        extra["kind"] = kind
    np.savez_compressed(
        fname,
        **feature_arrays,
        ligand_idx=np.int32(idx),
        ligand_name=name,
        source_sdf=str(sdf_path),
        source_pdb=str(pdb_path),
        pic50_pred=np.float32(model_affinity["pic50"]),
        pkd_pred=np.float32(model_affinity["pkd"]),
        pki_pred=np.float32(model_affinity["pki"]),
        pec50_pred=np.float32(model_affinity["pec50"]),
        noise_scale=np.float32(args.coord_noise_scale),
        **extra,
    )
    t_end = time.perf_counter()
    if stage_log is not None:
        stage_log.setdefault("validate_ms", []).append(
            (t_before_write - t_after_predict) * 1000.0
        )
        stage_log.setdefault("write_ms", []).append(
            (t_end - t_before_write) * 1000.0
        )
        stage_log.setdefault("total_ms", []).append((t_end - t_start) * 1000.0)

    return {
        "idx": idx,
        "name": name,
        "label": label,
        "kind": kind,
        "npz_path": str(fname),
        "elapsed_ms": elapsed_ms,
        "model_affinity": model_affinity,
        "manual_affinity": manual_affinity,
        "head_diffs": head_diffs,
        "concat_diff": concat_diff,
        "fire_counts": fire_counts_snapshot,
        "pass2_tensors": pass2,
    }


# ---------------------------------------------------------------------------
# CPU-side parallel system building (Phase 4 Step 4 — surrogate for DataLoader workers).
#
# load_data_from_pdb (process_complex) is CPU-only — it parses the PDB + ligand
# SDF, cuts the pocket, and builds a PocketComplex. For DUDE-Z primary it's
# ~70 ms/pose serially. Parallelizing across CPU cores via a spawn-context
# Pool drops this to ~load_data_ms / N_workers without touching CUDA in
# workers (no fork bug). Workers do NOT need GPU; they return CPU
# PocketComplex objects which pickle cleanly.
#
# Hard rule (brief §4 Step 4): no torch.cuda calls in workers. The worker
# function only imports FLOWR's CPU-side parsing path.
# ---------------------------------------------------------------------------

_WORKER_INITIALIZED = False


def _worker_init():
    """Spawn-context worker initializer — runs once per worker process.

    Pays the FLOWR import cost ONCE per worker, even if we reuse the pool
    across many targets (Phase 5 persistent-pool pattern). Without this,
    the first task per worker would pay the ~10s import; with this, all
    tasks see warm imports.
    """
    import sys
    for p in ("/home/maurbina/flowr_root", "/home/maurbina/hdbind-3D"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import warnings
    warnings.filterwarnings("ignore")
    # Pre-import the heavyweight modules so the first task runs warm.
    import flowr.gen.utils  # noqa: F401
    import flowr.data.preprocess_pdbs  # noqa: F401


def _worker_build_system(payload: dict):
    """Top-level (picklable) worker that builds one system on CPU.

    Returns the PocketComplex on success, or None if any failure (matches the
    silent-drop semantics of FLOWR's process_complex on bad inputs).
    """
    import sys
    for p in ("/home/maurbina/flowr_root", "/home/maurbina/hdbind-3D"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import warnings
    warnings.filterwarnings("ignore")
    from flowr.gen.utils import load_data_from_pdb
    from types import SimpleNamespace
    args = SimpleNamespace(**payload["args_vars"])
    try:
        sys_i = load_data_from_pdb(
            args,
            remove_hs=payload["remove_hs"],
            remove_aromaticity=payload["remove_aromaticity"],
            ligand_idx=payload["ligand_idx"],
        )
    except Exception:
        sys_i = None
    return sys_i


def _build_systems_parallel_perpose(
    args, hparams, records, n_workers: int, pool=None
) -> tuple[list, list[int]]:
    """Build N systems in parallel using PER-POSE single-mol SDFs.

    Each record carries a `tmp_path` (a single-mol SDF). Workers open that
    single-mol SDF directly — same input as Phase 3's serial path — so
    resulting systems are bit-identical to Phase 3 by construction.

    Why a separate path from `_build_systems_parallel`: that helper uses a
    shared multi-mol SDF with ligand_idx=i, which gives subtly different
    ligand parsing than per-pose SDFs (verified empirically — ligand-side
    features diverge ρ ≈ 0.8 vs Phase 3 even at batch size 1). The per-pose
    helper here preserves Phase 3's semantics.
    """
    import multiprocessing as mp
    base_vars = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    payloads = []
    for rec in records:
        v = dict(base_vars)
        v["ligand_file"] = str(rec["tmp_path"])
        v["multiple_ligands"] = False
        payloads.append({
            "args_vars": v,
            "remove_hs": hparams["remove_hs"],
            "remove_aromaticity": hparams["remove_aromaticity"],
            "ligand_idx": 0,
        })
    rec_to_sys: list[int] = []
    systems: list = []

    def _consume(p):
        for i, sys_i in enumerate(tqdm(
            p.imap(_worker_build_system, payloads, chunksize=8),
            total=len(payloads), desc=f"build[{n_workers}w,per-pose]", unit="sys",
            leave=False,
        )):
            if sys_i is None:
                rec_to_sys.append(-1)
                continue
            if isinstance(sys_i, list):
                sys_i = sys_i[0]
            rec_to_sys.append(len(systems))
            systems.append(sys_i)

    if pool is not None:
        _consume(pool)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_worker_init) as p:
            _consume(p)
    return systems, rec_to_sys


def _build_systems_parallel(
    args, hparams, records, n_workers: int
) -> tuple[list, list[int]]:
    """Build N systems in parallel. Returns (systems, rec_to_sys_index)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    args_vars = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    payloads = [
        {
            "args_vars": args_vars,
            "remove_hs": hparams["remove_hs"],
            "remove_aromaticity": hparams["remove_aromaticity"],
            "ligand_idx": i,
        }
        for i in range(len(records))
    ]
    # imap_unordered would be faster but loses ordering; we need ordering for
    # the records→systems mapping, so use ordered map.
    rec_to_sys: list[int] = []
    systems: list = []
    with ctx.Pool(n_workers) as pool:
        for i, sys_i in enumerate(tqdm(
            pool.imap(_worker_build_system, payloads, chunksize=8),
            total=len(payloads), desc=f"build[{n_workers}w]", unit="sys",
            leave=False,
        )):
            if sys_i is None:
                rec_to_sys.append(-1)
                continue
            if isinstance(sys_i, list):
                sys_i = sys_i[0]
            rec_to_sys.append(len(systems))
            systems.append(sys_i)
    return systems, rec_to_sys


# ---------------------------------------------------------------------------
# Batched (Phase 4 Step 3) — one multi-mol SDF, FLOWR multi-ligand codepath
# ---------------------------------------------------------------------------

def process_target_batched(
    target: str,
    pdb_path: Path,
    sdf_path: Path,
    records: list[dict],
    args_template: SimpleNamespace,
    model,
    hparams: dict,
    vocab,
    vocab_charges,
    vocab_hybridization,
    vocab_aromatic,
    transform,
    interpolant,
    hooks: HookCapture,
    output_dir: Path,
    batch_size: int,
    n_workers: int = 0,
    head_tolerance: float = 1e-3,
    concat_tolerance: float = 1e-5,
    stage_log: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run batched extraction over all of `records` (= all poses for one target).

    Builds N systems via load_data_from_pdb(ligand_idx=i), wraps them in one
    GeometricDataset (FLOWR's PocketComplexBatch path), iterates the batched
    dataloader, and on each batch captures pass-2 hook outputs as [B, D]
    tensors then slices per-pose features for .npz output.

    Per-batch validations (same as per-pose path, just vectorized):
      - Fire-count assertion: each hook fires exactly twice per batch (pass 1 + pass 2).
      - Algebraic: concat(z_*_post, dim=-1) == combined_at_pic50_head, per row.
      - Semantic: manual heads(combined) == predict_affinity_batch's returned
        affinity, per row. Same bit-exactness guarantee as Phase 2 / 3.

    Returns (results, failures).
    """
    args = SimpleNamespace(**vars(args_template))
    args.pdb_file = str(pdb_path)
    args.ligand_file = str(sdf_path)
    args.multiple_ligands = True
    # Force deterministic batch sizing: bucket_cost_scale='constant' means
    # bucket_costs default to 1, so batch_size_per_bucket = batch_cost / 1
    # = batch_cost. The bucketer still groups by atom-count bin to minimize
    # padding waste, but each emitted batch has exactly batch_cost systems.
    args.bucket_cost_scale = "constant"
    args.batch_cost = batch_size

    # ---- Build N systems ----
    t_build_start = time.perf_counter()
    if n_workers > 0:
        # CPU parallel build (Phase 4 Step 4 surrogate).
        a = SimpleNamespace(**vars(args))
        systems, rec_to_sys = _build_systems_parallel(a, hparams, records, n_workers)
    else:
        systems = []
        rec_to_sys = []
        for i, rec in enumerate(tqdm(records, desc="build", unit="sys", leave=False)):
            a = SimpleNamespace(**vars(args))
            a.pdb_id = f"lig_{rec['idx']:05d}"
            try:
                sys_i = load_data_from_pdb(
                    a,
                    remove_hs=hparams["remove_hs"],
                    remove_aromaticity=hparams["remove_aromaticity"],
                    ligand_idx=i,
                )
            except Exception:
                sys_i = None
            if sys_i is None:
                rec_to_sys.append(-1)
                continue
            if isinstance(sys_i, list):
                sys_i = sys_i[0]
            rec_to_sys.append(len(systems))
            systems.append(sys_i)
    t_after_build = time.perf_counter()

    dataset = get_dataset(systems, transform, vocab, interpolant, args, hparams)
    dataloader = get_dataloader(args, dataset, interpolant)
    t_after_loader = time.perf_counter()

    # Map sys_idx → record_idx for output naming
    sys_to_rec: list[int] = [-1] * len(systems)
    for r_i, s_i in enumerate(rec_to_sys):
        if s_i >= 0:
            sys_to_rec[s_i] = r_i

    results: list[dict] = []
    failures: list[dict] = []
    pose_pos = 0  # cursor over systems (which is also the dataloader order)

    pbar = tqdm(total=len(systems), desc=f"extract[B≤{batch_size}]", unit="lig", leave=False)
    for batch in dataloader:
        prior, posterior, _, _ = batch
        B = posterior["coords"].size(0)

        t_batch_start = time.perf_counter()
        hooks.reset()
        gen_ligs = predict_affinity_batch(
            args, model=model, prior=prior, posterior=posterior,
            noise_scale=args.coord_noise_scale, eps=1e-4, seed=args.seed,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_after_predict = time.perf_counter()

        # Fire-count: per batch each hook fires exactly twice (pass1 + pass2)
        hooks.assert_fired(expected=EXPECTED_FIRES_PER_HOOK)
        fire_counts_snapshot = dict(hooks.fire_counts)
        pass2 = hooks.pass2()

        # Per-row validation + slicing + .npz write
        expected_shapes = {
            "f_lig_pre": 1024, "f_pocket_pre": 512, "eij_pooled_pre": 128,
            "z_lig_post": 128, "z_pocket_post": 128, "z_int_post": 128,
            "combined_at_pic50_head": 384,
        }
        for k, d in expected_shapes.items():
            t = pass2[k]
            if t.dim() != 2 or t.shape != (B, d):
                raise AssertionError(
                    f"batched hook {k!r}: got shape {tuple(t.shape)} want ({B}, {d})"
                )
            if not torch.isfinite(t).all():
                raise AssertionError(f"batched hook {k!r} has NaN/Inf")

        concat_full = torch.cat(
            [pass2["z_lig_post"], pass2["z_pocket_post"], pass2["z_int_post"]],
            dim=-1,
        )
        combined_full = pass2["combined_at_pic50_head"]
        per_row_diff = (concat_full - combined_full).abs().amax(dim=-1)
        worst_concat = per_row_diff.max().item()
        if worst_concat > concat_tolerance:
            raise AssertionError(
                f"batched concat==combined check failed: worst |diff|={worst_concat:.3e} "
                f"(tol {concat_tolerance}); per_row_diff[:5]={per_row_diff[:5].tolist()}"
            )

        # Manual heads on the whole batch at once (cheaper than per-row)
        device = next(model.gen.ligand_dec.parameters()).device
        combined_dev = combined_full.to(device)
        with torch.no_grad():
            manual_full = {
                "pic50": model.gen.ligand_dec.pic50_head(combined_dev).squeeze(-1).cpu(),
                "pkd":   model.gen.ligand_dec.pkd_head(combined_dev).squeeze(-1).cpu(),
                "pki":   model.gen.ligand_dec.pki_head(combined_dev).squeeze(-1).cpu(),
                "pec50": model.gen.ligand_dec.pec50_head(combined_dev).squeeze(-1).cpu(),
            }

        t_after_validate = time.perf_counter()

        for b in range(B):
            sys_idx = pose_pos + b
            rec = records[sys_to_rec[sys_idx]]
            model_aff = _affinity_props_from_mol(gen_ligs[b])
            manual_aff = {k: float(v[b].item()) for k, v in manual_full.items()}
            head_diffs = {k: abs(model_aff[k] - manual_aff[k])
                          for k in ("pic50", "pkd", "pki", "pec50")}
            worst_head_diff = max(head_diffs.values())
            if worst_head_diff > head_tolerance:
                raise AssertionError(
                    f"batched lig {rec['idx']} ({rec['name']}): manual heads do "
                    f"not match predict_affinity_batch — diffs={head_diffs} "
                    f"(tol {head_tolerance})"
                )

            feature_arrays = {
                k: pass2[k][b].float().numpy().astype(np.float32) for k in HOOK_NAMES
            }
            label = rec["label"]; kind = rec["kind"]; name = rec["name"]; idx = rec["idx"]
            label_tag = f"a{label}" if label is not None else "u"
            fname = output_dir / f"{idx:05d}_{label_tag}_{_sanitize_filename(name)}.npz"
            extra: dict = {}
            if label is not None: extra["label"] = np.int32(label)
            if kind is not None:  extra["kind"] = kind
            np.savez_compressed(
                fname, **feature_arrays,
                ligand_idx=np.int32(idx), ligand_name=name,
                source_sdf=str(sdf_path), source_pdb=str(pdb_path),
                pic50_pred=np.float32(model_aff["pic50"]),
                pkd_pred=np.float32(model_aff["pkd"]),
                pki_pred=np.float32(model_aff["pki"]),
                pec50_pred=np.float32(model_aff["pec50"]),
                noise_scale=np.float32(args.coord_noise_scale),
                **extra,
            )
            results.append({
                "idx": idx, "name": name, "label": label, "kind": kind,
                "npz_path": str(fname),
                "elapsed_ms": (t_after_predict - t_batch_start) * 1000.0 / B,
                "model_affinity": model_aff,
                "manual_affinity": manual_aff,
                "head_diffs": head_diffs,
                "concat_diff": float(per_row_diff[b].item()),
                "fire_counts": fire_counts_snapshot,
                # Store pass2 tensors only for the first pose (for stats printing)
                "pass2_tensors": {k: pass2[k][b:b+1] for k in HOOK_NAMES} if b == 0 and pose_pos == 0 else None,
            })

        t_after_write = time.perf_counter()
        if stage_log is not None:
            stage_log.setdefault("predict_ms", []).append(
                (t_after_predict - t_batch_start) * 1000.0 / B
            )
            stage_log.setdefault("validate_ms", []).append(
                (t_after_validate - t_after_predict) * 1000.0 / B
            )
            stage_log.setdefault("write_ms", []).append(
                (t_after_write - t_after_validate) * 1000.0 / B
            )
            stage_log.setdefault("total_ms", []).append(
                (t_after_write - t_batch_start) * 1000.0 / B
            )
            stage_log.setdefault("batch_size_actual", []).append(B)

        pose_pos += B
        pbar.update(B)
    pbar.close()

    # Account for skipped/None systems
    for r_i, s_i in enumerate(rec_to_sys):
        if s_i == -1:
            failures.append({
                "idx": records[r_i]["idx"], "name": records[r_i]["name"],
                "label": records[r_i]["label"], "kind": records[r_i]["kind"],
                "error": "load_data_from_pdb returned None (silent drop)",
            })

    # One-shot stage entries for build/loader (per pose, amortized)
    if stage_log is not None:
        total_poses = max(len(systems), 1)
        stage_log.setdefault("load_data_ms", []).append(
            (t_after_build - t_build_start) * 1000.0 / total_poses
        )
        stage_log.setdefault("dataloader_ms", []).append(
            (t_after_loader - t_after_build) * 1000.0 / total_poses
        )
    return results, failures


# ---------------------------------------------------------------------------
# Stats printer
# ---------------------------------------------------------------------------

def _print_feature_stats(pass2: dict[str, torch.Tensor], ligand_name: str) -> None:
    print(f"\n=== Feature stats for first ligand ({ligand_name}) ===")
    print(f"{'tensor':>26s}  {'shape':>12s}  {'min':>10s}  {'max':>10s}  {'mean':>10s}  {'std':>10s}")
    for k in HOOK_NAMES:
        t = pass2[k]
        shape = "x".join(str(s) for s in t.shape)
        print(
            f"{k:>26s}  {shape:>12s}  "
            f"{t.min().item():>+10.4f}  {t.max().item():>+10.4f}  "
            f"{t.mean().item():>+10.4f}  {t.std().item():>10.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_inputs(args_cli) -> tuple[str, Path, list[dict], int, Path | None]:
    """Resolve input mode (SDF or target) into a uniform records list.

    Returns (mode_label, pdb_path, records, raw_count, multi_sdf_or_None).
    For target mode at batch_size > 1, also returns a Path to the multi-mol SDF
    (one per target) that the batched codepath consumes. For per-pose mode the
    records carry individual tmp_paths.
    """
    if args_cli.target is not None:
        target = args_cli.target.upper()
        pdb_path = (DOCKING_DIR / target / "rec.crg.pdb").resolve()
        assert pdb_path.exists(), f"DUDE-Z receptor not found: {pdb_path}"
        # Raw counts via grep on MOLECULE blocks — matches phase3_mol2_audit.md.
        raw_count = 0
        for kind in ("ligand", "decoy"):
            from config import MOL2_SUBDIR, MOL2_PREFIX
            mol2 = DOCKING_DIR / target / MOL2_SUBDIR / f"{MOL2_PREFIX}_{kind}_poses.mol2"
            if mol2.exists():
                with open(mol2) as f:
                    raw_count += sum(1 for line in f if line.startswith("@<TRIPOS>MOLECULE"))
        # Phase 4 finding (see throughput report): the multi-mol SDF path
        # (--batch_size > 1) produces ligand-side features that drift from
        # Phase 3 (per-pose SDF) baseline at ρ ≈ 0.8. Use per-pose SDFs
        # whenever bit-correctness vs Phase 3 matters. Workers can still
        # parallelize the per-pose build CPU-side without going through the
        # multi-mol SDF path.
        if args_cli.batch_size > 1:
            sdf_path, recs = enumerate_target_poses_one_sdf(target)
            return f"target={target} batched B={args_cli.batch_size} W={args_cli.workers}", pdb_path, recs, raw_count, sdf_path
        recs = enumerate_target_poses(target)
        mode = f"target={target} per-pose"
        if args_cli.workers > 0:
            mode += f" W={args_cli.workers}"
        return mode, pdb_path, recs, raw_count, None
    else:
        pdb_path = Path(args_cli.pdb).resolve()
        sdf_path = Path(args_cli.ligand_file).resolve()
        assert pdb_path.exists(), f"PDB not found: {pdb_path}"
        assert sdf_path.exists(), f"Ligand SDF not found: {sdf_path}"
        raw_count = sum(
            1 for _ in Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        )
        # Adapt the SDF splitter's tuple output to the unified dict format.
        recs = []
        for idx, name, tmp_path, recovered in split_sdf_to_single_ligand_files(sdf_path):
            recs.append({
                "idx": idx, "name": name, "tmp_path": tmp_path,
                "label": None, "kind": None, "recovered": recovered,
            })
        return f"sdf={sdf_path}", pdb_path, recs, raw_count, None


def _per_target_extract(
    target: str, args_cli, args_template, model, hparams,
    vocab, vocab_charges, vocab_hybridization, vocab_aromatic,
    transform, interpolant, hooks, autocast_ctx, pool, hb_state,
):
    """Run extraction for one target inside the all-targets loop.

    Reuses the persistent worker pool, hooks, and model. Writes per-target
    .npz files + manifest under features_root/<target>/. Returns the
    per-target manifest dict.
    """
    # Convention: prefer ~/hdbind-3D/patched_inputs/<TARGET>/rec.crg.pdb over the
    # read-only upstream DUD-Z file if present. See notes/phase5_urok_patch.md.
    patched_pdb = _HDBIND / "patched_inputs" / target / "rec.crg.pdb"
    if patched_pdb.exists():
        pdb_path = patched_pdb.resolve()
        tqdm.write(f"  [patched input] {target}: using {pdb_path}")
    else:
        pdb_path = (DOCKING_DIR / target / "rec.crg.pdb").resolve()
    assert pdb_path.exists(), f"DUDE-Z receptor not found: {pdb_path}"
    target_dir = (Path(args_cli.features_root) / target).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Raw MOLECULE count for ground-truth check
    raw_count = 0
    for kind in ("ligand", "decoy"):
        from config import MOL2_SUBDIR, MOL2_PREFIX
        mol2 = DOCKING_DIR / target / MOL2_SUBDIR / f"{MOL2_PREFIX}_{kind}_poses.mol2"
        if mol2.exists():
            with open(mol2) as f:
                raw_count += sum(1 for line in f if line.startswith("@<TRIPOS>MOLECULE"))

    records = enumerate_target_poses(target)
    hb_state.update(poses_total=len(records), poses_done=0)

    # Parallel CPU prebuild using the persistent pool
    t_pb_start = time.perf_counter()
    a = SimpleNamespace(**vars(args_template))
    a.pdb_file = str(pdb_path)
    a.multiple_ligands = False
    sys_list, rec_to_sys = _build_systems_parallel_perpose(
        a, hparams, records, args_cli.workers, pool=pool
    )
    t_pb_done = time.perf_counter()
    prebuilt_systems = {
        records[ri]["idx"]: sys_list[si]
        for ri, si in enumerate(rec_to_sys) if si >= 0
    }
    pb_drops = sum(1 for si in rec_to_sys if si < 0)

    # Serial GPU forward, B=1
    results: list[dict] = []
    failures: list[dict] = []
    t_target_start = time.perf_counter()
    gpu_peak_bytes = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inner = tqdm(records, desc=f"{target}", unit="lig", position=1, leave=False)
    for rec in inner:
        try:
            with autocast_ctx:
                res = process_one_ligand(
                    idx=rec["idx"], name=rec["name"],
                    sdf_path=rec["tmp_path"], pdb_path=pdb_path,
                    args_template=args_template, model=model, hparams=hparams,
                    vocab=vocab, vocab_charges=vocab_charges,
                    vocab_hybridization=vocab_hybridization, vocab_aromatic=vocab_aromatic,
                    transform=transform, interpolant=interpolant,
                    hooks=hooks, output_dir=target_dir,
                    label=rec["label"], kind=rec["kind"],
                    prebuilt_system=prebuilt_systems.get(rec["idx"]),
                )
            results.append(res)
            hb_state.update(poses_done=len(results))
            if torch.cuda.is_available():
                gpu_peak_bytes = max(gpu_peak_bytes, torch.cuda.max_memory_allocated())
        except Exception as exc:
            failures.append({
                "idx": rec["idx"], "name": rec["name"],
                "label": rec["label"], "kind": rec["kind"],
                "error": repr(exc),
            })
            tqdm.write(f"FAILED target={target} idx={rec['idx']} name={rec['name']}: {exc!r}")
        finally:
            try:
                rec["tmp_path"].unlink(missing_ok=True)
            except Exception:
                pass
    inner.close()

    elapsed_total = time.perf_counter() - t_target_start
    prebuild_time = t_pb_done - t_pb_start

    # Account for prebuild silent drops
    for ri, si in enumerate(rec_to_sys):
        if si < 0:
            failures.append({
                "idx": records[ri]["idx"], "name": records[ri]["name"],
                "label": records[ri]["label"], "kind": records[ri]["kind"],
                "error": "load_data_from_pdb returned None during parallel prebuild",
            })

    # Manifest — keyed to target for clean Threadripper merge
    import socket
    manifest = {
        "target": target,
        "extraction_host": socket.gethostname(),
        "extraction_device": str(next(model.parameters()).device),
        "extraction_timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "ckpt": str(args_cli.ckpt_path),
        "config": {
            "batch_size": args_cli.batch_size,
            "workers": args_cli.workers,
            "bf16": args_cli.bf16,
            "noise_scale": args_cli.coord_noise_scale,
            "pocket_cutoff": args_cli.pocket_cutoff,
            "seed": args_cli.seed,
        },
        "source_pdb": str(pdb_path),
        "n_raw_in_source": raw_count,
        "n_readable": len(records),
        "n_extracted": len(results),
        "n_failed": len(failures),
        "n_actives": sum(1 for r in results if r.get("label") == 1),
        "n_decoys": sum(1 for r in results if r.get("label") == 0),
        "failures": failures,
        "feature_dims": {
            "f_lig_pre": 1024, "f_pocket_pre": 512, "eij_pooled_pre": 128,
            "z_lig_post": 128, "z_pocket_post": 128, "z_int_post": 128,
            "combined_at_pic50_head": 384,
        },
        "validation": {
            "fire_count_per_hook_expected": EXPECTED_FIRES_PER_HOOK,
            "fire_count_assertion": "passed" if all(
                all(c == EXPECTED_FIRES_PER_HOOK for c in r["fire_counts"].values())
                for r in results
            ) else "FAILED",
            "concat_eq_combined": {
                "passed": (max((r["concat_diff"] for r in results), default=0.0) <= 1e-5),
                "worst_abs_diff": max((r["concat_diff"] for r in results), default=0.0),
                "tolerance": 1e-5,
            },
            "manual_head_eq_predict": {
                "passed": (max((max(r["head_diffs"].values()) for r in results), default=0.0) <= 1e-3),
                "worst_abs_diff": max((max(r["head_diffs"].values()) for r in results), default=0.0),
                "tolerance": 1e-3,
            },
        },
        "elapsed_seconds": elapsed_total,
        "prebuild_seconds": prebuild_time,
        "ms_per_ligand_avg": (elapsed_total + prebuild_time) * 1000.0 / max(len(results), 1),
        "gpu_peak_memory_gb": gpu_peak_bytes / 1e9,
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _main_all_targets(args_cli):
    """Phase 5 entrypoint: extract all 43 DUDE-Z primary targets."""
    features_root = Path(args_cli.features_root).resolve()
    features_root.mkdir(parents=True, exist_ok=True)

    all_targets = _list_all_dudez_targets()
    if not all_targets:
        raise SystemExit(f"no targets found under {DOCKING_DIR}")
    print(f"All DUDE-Z primary targets: {len(all_targets)}")

    # Resume filter
    skipped_complete: list[tuple[str, int]] = []
    targets_to_run: list[str] = []
    for t in all_targets:
        ok, m = _target_is_complete(t, features_root) if not args_cli.no_resume else (False, None)
        if ok:
            skipped_complete.append((t, m["n_extracted"]))
        else:
            targets_to_run.append(t)
    if skipped_complete:
        print(f"--resume: skipping {len(skipped_complete)} complete targets:")
        for t, n in skipped_complete:
            print(f"    {t}: {n} extracted")
    print(f"\nTargets to run: {len(targets_to_run)}")
    print(f"  {', '.join(targets_to_run[:10])}{' ...' if len(targets_to_run) > 10 else ''}")

    if not targets_to_run:
        print("Nothing to do — all targets already complete.")
        return

    print(f"\nConfig: --workers {args_cli.workers}  --batch_size {args_cli.batch_size}  "
          f"--bf16={args_cli.bf16}  --features_root {features_root}")
    print(f"Heartbeat: every {args_cli.heartbeat_s}s\n")

    torch.set_float32_matmul_precision("high")
    if args_cli.bf16:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args_template = make_base_args(
        args_cli.ckpt_path,
        str(features_root),
        num_workers=0,
        seed=args_cli.seed,
        batch_cost=args_cli.batch_cost,
        coord_noise_scale=args_cli.coord_noise_scale,
        pocket_cutoff=args_cli.pocket_cutoff,
    )

    # Load model ONCE
    print("=== Loading model ===")
    t_model_start = time.perf_counter()
    (model, hparams, vocab, vocab_charges, vocab_hybridization, vocab_aromatic,
     vocab_pocket_atoms, vocab_pocket_res) = load_model(args_template)
    devs = [d.strip() for d in args_cli.devices.split(",") if d.strip()]
    if len(devs) > 1:
        print(f"--devices {args_cli.devices}: scaffold-only; using {devs[0]} only.")
    device = torch.device(devs[0] if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"device: {device}   model load: {time.perf_counter()-t_model_start:.1f}s")

    transform, interpolant = load_util(
        args_template, hparams, vocab, vocab_charges, vocab_hybridization, vocab_aromatic
    )

    # Persistent worker pool — paid ONCE, reused across all targets
    import multiprocessing as mp
    print(f"\n=== Spawning {args_cli.workers} persistent CPU workers ===")
    t_pool_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args_cli.workers, initializer=_worker_init)
    # Warm workers — submit a no-op so each child completes _worker_init before
    # the first real task starts. Reports stable per-target throughput.
    list(pool.imap(_noop_warmup, range(args_cli.workers)))
    print(f"worker pool ready: {time.perf_counter()-t_pool_start:.1f}s")

    # Hooks
    hooks = HookCapture(model.gen.ligand_dec)
    hooks.register()
    print(f"hooks: {HOOK_NAMES}")

    # Autocast
    if args_cli.bf16:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    # Heartbeat
    hb_state = _HeartbeatState()
    hb_state.update(targets_total=len(targets_to_run))
    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(hb_state, args_cli.heartbeat_s), daemon=True
    )
    hb_thread.start()

    # Main loop
    per_target_results: list[dict] = []
    t_total_start = time.perf_counter()
    outer = tqdm(targets_to_run, desc="targets", unit="t", position=0, leave=True)
    for ti, target in enumerate(outer):
        hb_state.update(current_target=target, targets_done=ti, poses_done=0, poses_total=0)
        outer.set_description(f"targets ({target})")
        try:
            m = _per_target_extract(
                target=target, args_cli=args_cli, args_template=args_template,
                model=model, hparams=hparams,
                vocab=vocab, vocab_charges=vocab_charges,
                vocab_hybridization=vocab_hybridization, vocab_aromatic=vocab_aromatic,
                transform=transform, interpolant=interpolant,
                hooks=hooks, autocast_ctx=autocast_ctx, pool=pool, hb_state=hb_state,
            )
            per_target_results.append(m)
            poses_done_total = sum(r["n_extracted"] for r in per_target_results)
            ms_per_lig_avg = (
                sum(r["elapsed_seconds"] + r.get("prebuild_seconds", 0)
                    for r in per_target_results) * 1000.0
                / max(poses_done_total, 1)
            )
            gpu_gb = max(r.get("gpu_peak_memory_gb", 0) for r in per_target_results)
            elapsed_so_far = time.perf_counter() - t_total_start
            remaining_targets = len(targets_to_run) - ti - 1
            eta_min = (elapsed_so_far / max(ti + 1, 1)) * remaining_targets / 60.0
            outer.set_postfix({
                "poses": poses_done_total,
                "ms/pose": f"{ms_per_lig_avg:.1f}",
                "gpu_gb": f"{gpu_gb:.1f}",
                "eta_min": f"{eta_min:.0f}",
            })
            tqdm.write(
                f"  {target}: n={m['n_extracted']}/{m['n_raw_in_source']} "
                f"failed={m['n_failed']} "
                f"elapsed={m['elapsed_seconds']:.1f}s "
                f"prebuild={m.get('prebuild_seconds', 0):.1f}s "
                f"ms/lig={m['ms_per_ligand_avg']:.1f}"
            )
        except Exception as exc:
            tqdm.write(f"TARGET FAILED {target}: {exc!r}")
            import traceback
            tqdm.write(traceback.format_exc())
    outer.close()
    hb_state.update(stop=True)
    hb_thread.join(timeout=5)

    pool.close()
    pool.join()
    hooks.remove()

    elapsed_total = time.perf_counter() - t_total_start

    # Final per-target throughput table
    print(f"\n\n=== Scenario A summary ({len(per_target_results)} targets, "
          f"{elapsed_total/60:.1f} min wall-clock) ===\n")
    header = ("Target", "Poses", "Failed", "Time(s)", "Prebuild(s)", "ms/pose", "GPU peak (GB)")
    print(f"{header[0]:<10s}  {header[1]:>6s}  {header[2]:>6s}  "
          f"{header[3]:>9s}  {header[4]:>10s}  {header[5]:>8s}  {header[6]:>12s}")
    total_poses = 0
    total_failed = 0
    for m in per_target_results:
        print(
            f"{m['target']:<10s}  "
            f"{m['n_extracted']:>6d}  "
            f"{m['n_failed']:>6d}  "
            f"{m['elapsed_seconds']:>9.1f}  "
            f"{m.get('prebuild_seconds', 0):>10.1f}  "
            f"{m['ms_per_ligand_avg']:>8.1f}  "
            f"{m.get('gpu_peak_memory_gb', 0):>12.2f}"
        )
        total_poses += m["n_extracted"]
        total_failed += m["n_failed"]
    print(f"\nTOTAL: {len(per_target_results)} targets, {total_poses} poses extracted, "
          f"{total_failed} failed, {elapsed_total/3600:.2f} hours "
          f"({total_poses / elapsed_total:.1f} poses/sec)")


def _noop_warmup(_):
    """Tiny task to force each persistent worker to run its initializer."""
    return None


def main(args_cli):
    if args_cli.all_targets:
        return _main_all_targets(args_cli)
    mode_label, pdb_path, records, raw_count, multi_sdf = _resolve_inputs(args_cli)

    output_dir = Path(args_cli.output_dir).resolve() if args_cli.output_dir else None
    if output_dir is None and args_cli.target:
        output_dir = (_HDBIND / "features" / "flowr_root" / args_cli.target.upper()).resolve()
    if output_dir is None:
        raise SystemExit("--output_dir is required (no default for SDF mode)")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"mode:    {mode_label}")
    print(f"PDB:     {pdb_path}")
    print(f"Output:  {output_dir}")
    print(f"Ckpt:    {args_cli.ckpt_path}")
    print(f"noise_scale = {args_cli.coord_noise_scale}  (0.0 = clean-pose extraction)")

    torch.set_float32_matmul_precision("high")
    if args_cli.bf16:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Phase 4 Step 5: BF16 autocast + TF32 enabled")
    args_template = make_base_args(
        args_cli.ckpt_path,
        str(output_dir),
        num_workers=0,
        seed=args_cli.seed,
        batch_cost=args_cli.batch_cost,
        coord_noise_scale=args_cli.coord_noise_scale,
        pocket_cutoff=args_cli.pocket_cutoff,
    )

    print("\n=== Loading model ===")
    (model, hparams, vocab, vocab_charges, vocab_hybridization, vocab_aromatic,
     vocab_pocket_atoms, vocab_pocket_res) = load_model(args_template)
    # Phase 4 Step 8 scaffold: honor first device from --devices. Multi-GPU
    # target-level sharding (one device per spawned process) requires a
    # separate runner script and is out of scope for the slow debug machine.
    devs = [d.strip() for d in args_cli.devices.split(",") if d.strip()]
    if len(devs) > 1:
        print(f"--devices {args_cli.devices}: scaffold-only on this machine. "
              f"Using only {devs[0]}; rest reserved for high-perf transfer.")
    device = torch.device(devs[0] if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"device: {device}   model: {type(model).__name__}")

    transform, interpolant = load_util(
        args_template, hparams, vocab, vocab_charges, vocab_hybridization, vocab_aromatic
    )

    print("\n=== Registering hooks (6 production + 1 validation) ===")
    hooks = HookCapture(model.gen.ligand_dec)
    hooks.register()
    print(f"hooks: {HOOK_NAMES}")

    pocket_cache: PocketEncoderCache | None = None
    if args_cli.pocket_cache:
        pocket_cache = PocketEncoderCache(model.gen.pocket_enc)
        pocket_cache.install()
        print("pocket-encoder cache: INSTALLED (Phase 4 Step 2)")

    # Verify label counts vs ground truth for target mode (Phase 3 requirement).
    if args_cli.target is not None:
        from collections import Counter
        kind_counts = Counter(r["kind"] for r in records)
        print(f"\n=== Label counts (target={args_cli.target}) ===")
        print(f"  raw MOLECULE blocks in mol2: {raw_count}")
        print(f"  parsed actives (label=1):    {kind_counts.get('ligand', 0)}")
        print(f"  parsed decoys (label=0):     {kind_counts.get('decoy', 0)}")
        print(f"  total parsed:                {len(records)}")
        if raw_count != len(records):
            print(f"  ⚠️  {raw_count - len(records)} silent RDKit sanitize drops "
                  f"(see notes/phase3_mol2_audit.md for per-target breakdown)")
    else:
        print(f"\n=== Splitting SDF into single-ligand inputs ===")
        print(f"input ligands: {raw_count} ; readable: {len(records)}")
        rescued = [r for r in records if r["recovered"]]
        if rescued:
            print(f"recovered via _fix_valence: {len(rescued)}")

    if getattr(args_cli, "limit", None):
        if multi_sdf is None:
            for r in records[args_cli.limit:]:
                try:
                    r["tmp_path"].unlink(missing_ok=True)
                except Exception:
                    pass
        records = records[:args_cli.limit]
        print(f"  --limit {args_cli.limit}: truncated record list to {len(records)}")

    results: list[dict] = []
    failures: list[dict] = []
    stage_log: dict[str, list[float]] = {}
    gpu_mem_peak_bytes = 0
    t_total = time.perf_counter()

    # Phase 4 Step 5: autocast wrapping. Inner forward calls inherit the context.
    if args_cli.bf16:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext()

    # ── Branch: batched (Step 3) vs per-pose ─────────────────────────────────
    # Note: autocast_ctx (BF16 if --bf16, nullcontext otherwise) wraps both
    # the model forward AND the manual head replay inside process_target_batched,
    # so the semantic-equality check remains valid under BF16.
    if multi_sdf is not None:
        # Truncate multi-SDF to first N if --limit (build a smaller SDF)
        if args_cli.limit is not None and args_cli.limit < len(records):
            limited_sdf = Path(tempfile.NamedTemporaryFile(suffix=".sdf", delete=False).name)
            suppl = Chem.SDMolSupplier(str(multi_sdf), removeHs=False, sanitize=True)
            writer = Chem.SDWriter(str(limited_sdf))
            for i, m in enumerate(suppl):
                if i >= args_cli.limit:
                    break
                if m is not None:
                    writer.write(m)
            writer.close()
            try:
                multi_sdf.unlink(missing_ok=True)
            except Exception:
                pass
            multi_sdf = limited_sdf

        try:
            with autocast_ctx:
                results, failures = process_target_batched(
                    target=args_cli.target,
                    pdb_path=pdb_path,
                    sdf_path=multi_sdf,
                    records=records,
                    args_template=args_template, model=model, hparams=hparams,
                    vocab=vocab, vocab_charges=vocab_charges,
                    vocab_hybridization=vocab_hybridization, vocab_aromatic=vocab_aromatic,
                    transform=transform, interpolant=interpolant,
                    hooks=hooks, output_dir=output_dir,
                    batch_size=args_cli.batch_size,
                    n_workers=args_cli.workers,
                    stage_log=stage_log if args_cli.instrument else None,
                )
            if torch.cuda.is_available():
                gpu_mem_peak_bytes = torch.cuda.max_memory_allocated()
            # Print stats for first pose
            first = next((r for r in results if r.get("pass2_tensors")), None)
            if first is not None:
                _print_feature_stats(first["pass2_tensors"], first["name"])
        finally:
            try:
                multi_sdf.unlink(missing_ok=True)
            except Exception:
                pass
    else:
        # Optionally pre-build all systems in parallel from per-pose SDFs.
        prebuilt_systems: dict[int, object] = {}
        if args_cli.workers > 0 and args_cli.target is not None:
            a = SimpleNamespace(**vars(args_template))
            a.pdb_file = str(pdb_path)
            a.multiple_ligands = False
            t_pb_start = time.perf_counter()
            sys_list, rec_to_sys = _build_systems_parallel_perpose(
                a, hparams, records, args_cli.workers
            )
            t_pb_done = time.perf_counter()
            print(f"\nparallel-prebuild: {len(sys_list)} / {len(records)} systems "
                  f"in {t_pb_done - t_pb_start:.1f}s "
                  f"({(t_pb_done - t_pb_start)*1000/max(len(records),1):.1f} ms/lig)")
            for ri, si in enumerate(rec_to_sys):
                if si >= 0:
                    prebuilt_systems[records[ri]["idx"]] = sys_list[si]
            if stage_log is not None:
                stage_log.setdefault("load_data_ms", []).append(
                    (t_pb_done - t_pb_start) * 1000.0 / max(len(records), 1)
                )

        for split_pos, rec in enumerate(tqdm(records, desc="extract", unit="lig")):
            orig_idx = rec["idx"]
            name = rec["name"]
            tmp_sdf = rec["tmp_path"]
            try:
                with autocast_ctx:
                    res = process_one_ligand(
                        idx=orig_idx, name=name, sdf_path=tmp_sdf, pdb_path=pdb_path,
                        args_template=args_template, model=model, hparams=hparams,
                        vocab=vocab, vocab_charges=vocab_charges,
                        vocab_hybridization=vocab_hybridization, vocab_aromatic=vocab_aromatic,
                        transform=transform, interpolant=interpolant,
                        hooks=hooks, output_dir=output_dir,
                        label=rec["label"], kind=rec["kind"],
                        stage_log=stage_log if args_cli.instrument else None,
                        prebuilt_system=prebuilt_systems.get(orig_idx),
                    )
                if torch.cuda.is_available():
                    gpu_mem_peak_bytes = max(
                        gpu_mem_peak_bytes, torch.cuda.max_memory_allocated()
                    )
                results.append(res)
                if split_pos == 0:
                    _print_feature_stats(res["pass2_tensors"], res["name"])
            except Exception as exc:
                failures.append({
                    "idx": orig_idx, "name": name,
                    "label": rec["label"], "kind": rec["kind"],
                    "error": repr(exc),
                })
                tqdm.write(f"FAILED idx={orig_idx} name={name}: {exc!r}")
            finally:
                try:
                    tmp_sdf.unlink(missing_ok=True)
                except Exception:
                    pass

    elapsed_total = time.perf_counter() - t_total
    hooks.remove()
    if pocket_cache is not None:
        print(f"pocket-encoder cache stats: "
              f"{pocket_cache.n_hits} hits / {pocket_cache.n_misses} misses")
        pocket_cache.uninstall()

    # Manifest
    if results:
        head_diffs = [r["head_diffs"] for r in results]
        worst_head_diff = max(max(d.values()) for d in head_diffs)
        worst_concat_diff = max(r["concat_diff"] for r in results)
    else:
        worst_head_diff = worst_concat_diff = float("nan")

    manifest = {
        "mode": mode_label,
        "target": args_cli.target,
        "source_pdb": str(pdb_path),
        "source_sdf": (str(Path(args_cli.ligand_file).resolve())
                       if args_cli.ligand_file else None),
        "ckpt": str(args_cli.ckpt_path),
        "noise_scale": args_cli.coord_noise_scale,
        "pocket_cutoff": args_cli.pocket_cutoff,
        "seed": args_cli.seed,
        "n_raw_in_source": raw_count,
        "n_readable": len(records),
        "n_extracted": len(results),
        "n_failed": len(failures),
        "n_actives": sum(1 for r in results if r.get("label") == 1),
        "n_decoys": sum(1 for r in results if r.get("label") == 0),
        "failures": failures,
        "feature_dims": {
            "f_lig_pre": 1024, "f_pocket_pre": 512, "eij_pooled_pre": 128,
            "z_lig_post": 128, "z_pocket_post": 128, "z_int_post": 128,
            "combined_at_pic50_head": 384,
        },
        "validation": {
            "fire_count_per_hook_expected": EXPECTED_FIRES_PER_HOOK,
            "fire_count_assertion": "passed" if all(
                all(c == EXPECTED_FIRES_PER_HOOK for c in r["fire_counts"].values())
                for r in results
            ) else "FAILED",
            "concat_eq_combined": {
                "passed": worst_concat_diff <= 1e-5,
                "worst_abs_diff": worst_concat_diff,
                "tolerance": 1e-5,
            },
            "manual_head_eq_predict": {
                "passed": worst_head_diff <= 1e-3,
                "worst_abs_diff": worst_head_diff,
                "tolerance": 1e-3,
            },
        },
        "elapsed_seconds": elapsed_total,
        "ms_per_ligand_avg": (elapsed_total * 1000.0 / max(len(results), 1)),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if args_cli.instrument and stage_log:
        print(f"\n=== Per-stage breakdown (N={len(stage_log.get('total_ms', []))}) ===")
        print(f"{'stage':>16s}  {'mean_ms':>10s}  {'med_ms':>10s}  "
              f"{'p95_ms':>10s}  {'pct':>6s}")
        total_mean = float(np.mean(stage_log.get("total_ms", [0])))
        for stage in ("load_data_ms", "dataloader_ms", "predict_ms",
                      "validate_ms", "write_ms", "total_ms"):
            vals = stage_log.get(stage, [])
            if not vals:
                continue
            a = np.array(vals)
            pct = (a.mean() / total_mean * 100) if total_mean > 0 else 0
            print(f"{stage:>16s}  {a.mean():>10.2f}  {np.median(a):>10.2f}  "
                  f"{np.percentile(a, 95):>10.2f}  {pct:>5.1f}%")
        if torch.cuda.is_available():
            print(f"\nGPU peak memory: {gpu_mem_peak_bytes / 1e9:.2f} GB")
        # Stash in manifest
        manifest["stage_breakdown_ms"] = {
            s: {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)),
                "n": len(v),
            }
            for s, v in stage_log.items()
        }
        manifest["gpu_peak_memory_gb"] = gpu_mem_peak_bytes / 1e9
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n=== Done in {elapsed_total:.1f}s  "
          f"({manifest['ms_per_ligand_avg']:.1f} ms / lig) ===")
    print(f"  extracted:           {len(results)} / {len(records)} readable "
          f"({raw_count} raw)")
    if args_cli.target is not None:
        print(f"  actives / decoys:    {manifest['n_actives']} / {manifest['n_decoys']}")
    print(f"  failures:            {len(failures)}")
    print(f"  concat vs combined:  worst |diff| = {worst_concat_diff:.3e}  "
          f"(tol 1e-5) — {'OK' if worst_concat_diff <= 1e-5 else 'FAIL'}")
    print(f"  manual heads vs pred: worst |diff| = {worst_head_diff:.3e}  "
          f"(tol 1e-3) — {'OK' if worst_head_diff <= 1e-3 else 'FAIL'}")
    print(f"  manifest:            {manifest_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--target",
                     help="Phase 3 mode: DUDE-Z target name (e.g. AA2AR). "
                          "Iterates poses via src.data_loading.iter_poses() and "
                          "labels actives=1 / decoys=0 in the output .npz.")
    src.add_argument("--ligand_file",
                     help="Phase 2 mode: explicit SDF file (single or multi-ligand). "
                          "MOL2 not supported via this flag — use --target for mol2.")
    src.add_argument("--all_targets", action="store_true",
                     help="Phase 5 mode: extract all 43 DUDE-Z primary targets. "
                          "Loads model once, reuses worker pool across targets, "
                          "writes one .npz dir per target under --features_root, "
                          "and resumes any target whose manifest reports complete "
                          "coverage (unless --no_resume).")
    p.add_argument("--pdb",
                   help="protein PDB. REQUIRED with --ligand_file. "
                        "Auto-resolved to DOCKING_DIR/<TARGET>/rec.crg.pdb in --target mode.")
    p.add_argument("--output_dir",
                   help="output dir. REQUIRED with --ligand_file. "
                        "Defaults to ~/hdbind-3D/features/flowr_root/<TARGET>/ in --target mode.")
    p.add_argument("--ckpt_path", default=str(DEFAULT_CKPT))
    p.add_argument("--coord_noise_scale", type=float, default=0.0,
                   help="MUST be 0.0 for clean-pose feature extraction (default)")
    p.add_argument("--pocket_cutoff", type=float, default=7.0)
    p.add_argument("--batch_cost", type=int, default=20,
                   help="kept for parity with predict_aff*.sl; per-ligand loop "
                        "ignores batching")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="if set, process only the first N records (for quick sanity)")
    p.add_argument("--features_root",
                   default=str(_HDBIND / "features" / "flowr_root"),
                   help="root for per-target output subdirs (Phase 5 only). "
                        "Default: ~/hdbind-3D/features/flowr_root/")
    p.add_argument("--no_resume", action="store_true",
                   help="disable Phase 5 resume — re-extract all targets even "
                        "if their output dirs already have a complete manifest.")
    p.add_argument("--heartbeat_s", type=int, default=60,
                   help="Phase 5 heartbeat interval in seconds (default 60).")
    p.add_argument("--instrument", action="store_true",
                   help="capture per-stage timing (load_data / dataloader / "
                        "predict / validate / write) and print breakdown at end")
    p.add_argument("--pocket_cache", action="store_true",
                   help="Phase 4 Step 2: cache pocket_enc output once per target. "
                        "Safe within a target; reset between targets in Phase 5.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Phase 4 Step 3: batch N ligands per forward (target mode "
                        "only; falls back to per-pose loop in SDF mode). "
                        "Default 1 = no batching (Phase 3 behavior).")
    p.add_argument("--bf16", action="store_true",
                   help="Phase 4 Step 5: wrap model forward in BF16 autocast + "
                        "enable TF32. Validation expects features within BF16 "
                        "tolerance of FP32 (ρ > 0.999, not bit-exact).")
    p.add_argument("--workers", type=int, default=0,
                   help="Phase 4 Step 4: number of spawn-context CPU workers "
                        "for parallel load_data_from_pdb. 0 = serial build "
                        "(default). Recommend 8 on 32-core boxes.")
    p.add_argument("--devices", default="cuda:0",
                   help="Phase 4 Step 8 scaffold: comma-separated GPU list "
                        "(e.g. cuda:0,cuda:1,cuda:2,cuda:3). For now only the "
                        "FIRST device is honored; multi-GPU target sharding "
                        "via torch.multiprocessing.spawn will be activated "
                        "after transfer to the high-perf workstation.")
    args = p.parse_args()

    # Cross-flag validation
    if args.ligand_file is not None and args.pdb is None:
        p.error("--pdb is required with --ligand_file")
    if args.ligand_file is not None and args.output_dir is None:
        p.error("--output_dir is required with --ligand_file (no auto default)")
    # all_targets + workers > 0 is the only supported Phase 5 config
    if args.all_targets and args.workers <= 0:
        p.error("--all_targets requires --workers > 0 (recommend 16)")
    return args


if __name__ == "__main__":
    main(parse_args())
