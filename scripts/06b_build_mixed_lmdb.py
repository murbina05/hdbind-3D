"""
06b_build_mixed_lmdb.py — Build a 50/50-mixed-decoy training LMDB by sampling
from two existing single-source LMDBs.

TIER3_DECOY_BIAS_PLAN.md Phase 6 prereq: produce the AD-mixed training dataset
without re-featurizing. Both source LMDBs (DUDE-Z train + AD) were built with
identical parameters (pocket_cutoff=8.0, edge_cutoff=4.5, node_feature_dim=10,
edge_attr_dim=3, same element vocab) — verified against their manifests
before mixing.

Per-target mixing strategy: stratified by source.
  Per target T:
    n_per_source = min(N_dudez_decoys[T], N_ad_decoys[T])
    rng = np.random.default_rng(<RANDOM_SEED XOR md5(T)[:32]>)
    sample n_per_source DUDE-Z decoys + n_per_source AD decoys (no replacement)
    interleave alternating source for stable LMDB key order

Actives are read once from the DUDE-Z source and emitted unchanged
(actives in both source LMDBs are identical mol2 records pointing at the
same DOCKING_DIR/<TARGET>/DUDE_Z/ pose file).

`data.source ∈ {'dudez', 'ad', 'active-dudez'}` is recorded in the per-entry
metadata for downstream auditing; the training script ignores it.

Output layout (flat — Phase 6 training script reads <run_dir>/dataset.lmdb):
  outputs/06_build_egnn_dataset/<run_id>/
    dataset.lmdb
    index.csv      key,target,complex_id,label,source
    manifest.json  per-target counts + mixing config
    REPORT.md
    config.yaml
    build.log

Usage:
  python scripts/06b_build_mixed_lmdb.py \
    --dudez-dir outputs/06_build_egnn_dataset/20260505-000911-1662f0b \
    --ad-dir    outputs/06_build_egnn_dataset/all40-ad-20260508/ad \
    --run-id    all40-ad-mixed-20260508
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import PROJECT_ROOT, RANDOM_SEED
from src.data_loading import AD_EXCLUDED_TARGETS
from src.utils import setup_logging

log = logging.getLogger("06b_build_mixed_lmdb")

# Same map size as 06_build_egnn_dataset.py (large enough for any 40-target build).
LMDB_MAP_SIZE = 100 * 1024 ** 3

# Build-parameter fields that must match between the two source LMDBs.
PARAM_FIELDS = (
    "pocket_cutoff_ang", "edge_cutoff_ang",
    "node_feature_dim", "edge_attr_dim", "element_vocab",
)


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def _make_run_id(user_id: str | None) -> str:
    if user_id is not None:
        return user_id
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{_git_short_sha()}"


def _target_seed(target: str) -> int:
    """Stable per-target rng seed: RANDOM_SEED XOR md5(target)[:32 bits]."""
    h = hashlib.md5(target.upper().encode()).hexdigest()
    return RANDOM_SEED ^ int(h[:8], 16)


def _load_index(lmdb_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(lmdb_dir / "index.csv", dtype={"key": str})
    df["target"] = df["target"].str.upper()
    return df


def _verify_param_match(dudez_manifest: dict, ad_manifest: dict) -> None:
    mismatches = []
    for f in PARAM_FIELDS:
        a, b = dudez_manifest.get(f), ad_manifest.get(f)
        if a != b:
            mismatches.append((f, a, b))
    if mismatches:
        raise RuntimeError(
            "Source LMDBs were built with different parameters; mixing them "
            "would produce inconsistent features. Mismatches:\n  "
            + "\n  ".join(f"{f}: dudez={a!r} ad={b!r}" for f, a, b in mismatches)
        )


def _open_ro(lmdb_path: Path) -> lmdb.Environment:
    return lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False,
                     readahead=True, meminit=False, max_readers=512)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dudez-dir", type=Path, required=True,
                   help="Existing 06 run dir for DUDE-Z (has dataset.lmdb + index.csv).")
    p.add_argument("--ad-dir", type=Path, required=True,
                   help="Existing 06 run dir for AD (has dataset.lmdb + index.csv).")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: outputs/06_build_egnn_dataset/<run_id>/")
    p.add_argument("--ratio", type=float, default=0.5,
                   help="Fraction of decoys taken from AD per target (default 0.5 = 50/50).")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Restrict to these targets (default: AD targets, ABL1 excluded).")
    p.add_argument("--limit-per-target", type=int, default=None,
                   help="Cap total decoys per target after mixing (default: 2 × min sources).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.dudez_dir = args.dudez_dir.resolve()
    args.ad_dir = args.ad_dir.resolve()

    run_id = _make_run_id(args.run_id)
    if args.output_dir is None:
        args.output_dir = (PROJECT_ROOT / "outputs" / "06_build_egnn_dataset" / run_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=args.output_dir / "build.log")
    log.setLevel(logging.INFO)

    git_sha = _git_short_sha()
    log.info("=" * 70)
    log.info("06b_build_mixed_lmdb starting")
    log.info("  run_id: %s  | git_sha: %s", run_id, git_sha)
    log.info("  dudez_dir: %s", args.dudez_dir)
    log.info("  ad_dir:    %s", args.ad_dir)
    log.info("  ratio (AD fraction): %.3f", args.ratio)
    log.info("  output_dir: %s", args.output_dir)

    dudez_manifest = json.loads((args.dudez_dir / "manifest.json").read_text())
    ad_manifest = json.loads((args.ad_dir / "manifest.json").read_text())
    _verify_param_match(dudez_manifest, ad_manifest)
    log.info("  Build-parameter match verified (%s)", ", ".join(PARAM_FIELDS))

    dudez_idx = _load_index(args.dudez_dir)
    ad_idx = _load_index(args.ad_dir)

    # Determine target set.
    ad_targets = set(t for t in ad_idx["target"].unique()
                     if t not in AD_EXCLUDED_TARGETS)
    if args.targets:
        target_set = set(t.upper() for t in args.targets) & ad_targets
    else:
        target_set = ad_targets
    targets = sorted(target_set)
    log.info("  Targets to process: %d  (ABL1 excluded by default)", len(targets))

    # Open source LMDBs.
    dudez_env = _open_ro(args.dudez_dir / "dataset.lmdb")
    ad_env = _open_ro(args.ad_dir / "dataset.lmdb")

    # Open output LMDB.
    out_lmdb = args.output_dir / "dataset.lmdb"
    if out_lmdb.exists():
        out_lmdb.unlink()
    out_env = lmdb.open(str(out_lmdb), map_size=LMDB_MAP_SIZE, subdir=False, lock=False)

    next_key = 0
    index_rows: list[dict] = []
    per_target_counts: dict[str, dict] = {}

    with dudez_env.begin() as dz_txn, ad_env.begin() as ad_txn, out_env.begin(write=True) as out_txn:
        for target in targets:
            dz_active = dudez_idx[(dudez_idx.target == target) & (dudez_idx.label == 1)]
            dz_decoy = dudez_idx[(dudez_idx.target == target) & (dudez_idx.label == 0)]
            ad_decoy = ad_idx[(ad_idx.target == target) & (ad_idx.label == 0)]

            # Per-target balanced sampling.
            n_per_source = min(len(dz_decoy), len(ad_decoy))
            if args.limit_per_target is not None:
                n_per_source = min(n_per_source, args.limit_per_target // 2)
            if n_per_source == 0:
                log.warning("  %s: skipping (n_per_source=0; dz_decoys=%d ad_decoys=%d)",
                            target, len(dz_decoy), len(ad_decoy))
                per_target_counts[target] = {
                    "n_actives": 0, "n_dudez_decoys": 0, "n_ad_decoys": 0,
                    "n_per_source_target": 0, "n_per_source_actual": 0,
                    "skipped": "no decoys for one source",
                }
                continue

            rng = np.random.default_rng(_target_seed(target))
            dz_pick = rng.choice(dz_decoy["key"].values, size=n_per_source, replace=False)
            ad_pick = rng.choice(ad_decoy["key"].values, size=n_per_source, replace=False)

            # Emit actives first (from DUDE-Z), then interleaved decoys.
            for _, r in dz_active.iterrows():
                payload = dz_txn.get(r["key"].encode("ascii"))
                if payload is None:
                    log.warning("  %s: missing DUDE-Z active key %s", target, r["key"])
                    continue
                k = f"{next_key:09d}".encode("ascii")
                out_txn.put(k, payload)
                index_rows.append({
                    "key": k.decode(), "target": target, "complex_id": r["complex_id"],
                    "label": 1, "source": "active-dudez",
                })
                next_key += 1

            n_emitted_dz = 0
            n_emitted_ad = 0
            for i in range(n_per_source):
                # DUDE-Z decoy
                dz_k = dz_pick[i]
                payload = dz_txn.get(dz_k.encode("ascii"))
                if payload is None:
                    log.warning("  %s: missing DUDE-Z decoy key %s", target, dz_k)
                else:
                    cid = dz_decoy.set_index("key").loc[dz_k, "complex_id"]
                    k = f"{next_key:09d}".encode("ascii")
                    out_txn.put(k, payload)
                    index_rows.append({
                        "key": k.decode(), "target": target, "complex_id": cid,
                        "label": 0, "source": "dudez",
                    })
                    next_key += 1
                    n_emitted_dz += 1
                # AD decoy
                ad_k = ad_pick[i]
                payload = ad_txn.get(ad_k.encode("ascii"))
                if payload is None:
                    log.warning("  %s: missing AD decoy key %s", target, ad_k)
                else:
                    cid = ad_decoy.set_index("key").loc[ad_k, "complex_id"]
                    k = f"{next_key:09d}".encode("ascii")
                    out_txn.put(k, payload)
                    index_rows.append({
                        "key": k.decode(), "target": target, "complex_id": cid,
                        "label": 0, "source": "ad",
                    })
                    next_key += 1
                    n_emitted_ad += 1

            per_target_counts[target] = {
                "n_actives": int(len(dz_active)),
                "n_dudez_decoys_available": int(len(dz_decoy)),
                "n_ad_decoys_available": int(len(ad_decoy)),
                "n_per_source_target": n_per_source,
                "n_dudez_decoys_emitted": n_emitted_dz,
                "n_ad_decoys_emitted": n_emitted_ad,
            }
            log.info("  %s: actives=%d  dudez=%d/%d  ad=%d/%d  total=%d",
                     target, len(dz_active),
                     n_emitted_dz, len(dz_decoy),
                     n_emitted_ad, len(ad_decoy),
                     len(dz_active) + n_emitted_dz + n_emitted_ad)

    out_env.sync()
    out_env.close()
    dudez_env.close()
    ad_env.close()

    # ── Persist index, manifest, REPORT ───────────────────────────────────
    df_index = pd.DataFrame(index_rows)
    df_index.to_csv(args.output_dir / "index.csv", index=False)

    n_total = len(df_index)
    n_actives = int((df_index["label"] == 1).sum())
    n_decoys = n_total - n_actives
    n_dudez_decoys = int((df_index["source"] == "dudez").sum())
    n_ad_decoys = int((df_index["source"] == "ad").sum())

    log.info("─" * 60)
    log.info("Wrote %d entries: %d actives + %d decoys (%d dudez + %d ad)",
             n_total, n_actives, n_decoys, n_dudez_decoys, n_ad_decoys)

    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "venv": os.environ.get("VIRTUAL_ENV", "(none)"),
        "script": "scripts/06b_build_mixed_lmdb.py",
        "decoy_source": "dudez-ad-mix-50-50",
        "decoy_set": "DUDE_Z+AD-mix-50-50",   # legacy field for 11/12 string-match
        "mixing_strategy": "stratified-by-source per target",
        "mixing_strategy_notes": (
            "Per-target balanced subsample: n_per_source = "
            "min(N_dudez_decoys[T], N_ad_decoys[T]). RNG seed is "
            "RANDOM_SEED XOR md5(target)[:32 bits] for stable reproducibility. "
            "Decoys interleaved (one DUDE-Z, one AD, ...) for stable LMDB key order. "
            "Actives copied byte-identical from the DUDE-Z source."
        ),
        "ratio_ad": args.ratio,
        "limit_per_target": args.limit_per_target,
        "dudez_source": str(args.dudez_dir),
        "ad_source": str(args.ad_dir),
        "n_targets": len(per_target_counts),
        "n_complexes_total": n_total,
        "n_actives_total": n_actives,
        "n_decoys_total": n_decoys,
        "n_dudez_decoys_total": n_dudez_decoys,
        "n_ad_decoys_total": n_ad_decoys,
        "node_feature_dim": dudez_manifest["node_feature_dim"],
        "edge_attr_dim": dudez_manifest["edge_attr_dim"],
        "element_vocab": dudez_manifest["element_vocab"],
        "pocket_cutoff_ang": dudez_manifest["pocket_cutoff_ang"],
        "edge_cutoff_ang": dudez_manifest["edge_cutoff_ang"],
        "lmdb_path": "dataset.lmdb",
        "per_target": per_target_counts,
        "targets": sorted(per_target_counts.keys()),
        "ad_excluded_targets": sorted(AD_EXCLUDED_TARGETS),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    cfg = {"args": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}}
    cfg["args"]["targets"] = list(cfg["args"]["targets"]) if cfg["args"].get("targets") else None
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # REPORT.md
    lines = [
        "# 06b_build_mixed_lmdb — AD-mixed training LMDB",
        "",
        f"- run_id: `{run_id}`",
        f"- git_sha: `{git_sha}`",
        f"- decoy_source: **dudez-ad-mix-50-50**  (legacy decoy_set: `DUDE_Z+AD-mix-50-50`)",
        f"- mixing strategy: **stratified by source per target** (see manifest.mixing_strategy_notes)",
        f"- targets: {manifest['n_targets']}  (ABL1 excluded)",
        f"- LMDB entries: **{n_total}**  ({n_actives} actives + {n_decoys} decoys = "
        f"{n_dudez_decoys} dudez + {n_ad_decoys} ad)",
        f"- pocket_cutoff: {manifest['pocket_cutoff_ang']} Å, edge_cutoff: {manifest['edge_cutoff_ang']} Å",
        f"- sources: DUDE-Z `{args.dudez_dir}` ; AD `{args.ad_dir}`",
        "",
        "## Per-target",
        "",
        "| target | actives | dudez_avail | ad_avail | n_per_source | dudez_emit | ad_emit | total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in sorted(per_target_counts.keys()):
        c = per_target_counts[t]
        if c.get("skipped"):
            lines.append(f"| {t} | {c['n_actives']} | {c['n_dudez_decoys']} | {c['n_ad_decoys']} | "
                         f"0 | 0 | 0 | _skipped: {c['skipped']}_ |")
            continue
        total = c["n_actives"] + c["n_dudez_decoys_emitted"] + c["n_ad_decoys_emitted"]
        lines.append(
            f"| {t} | {c['n_actives']} | {c['n_dudez_decoys_available']} | "
            f"{c['n_ad_decoys_available']} | {c['n_per_source_target']} | "
            f"{c['n_dudez_decoys_emitted']} | {c['n_ad_decoys_emitted']} | {total} |"
        )
    lines += ["", "## Notes", "", "_Filled in after review by the operator._", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))

    log.info("Outputs at %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
