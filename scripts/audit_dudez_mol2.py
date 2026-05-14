#!/usr/bin/env python
"""Audit DUDE-Z primary mol2 inputs prior to Phase 3 feature extraction.

For each smoke target (AA2AR, ADRB2, EGFR by default):
  - Count raw MOLECULE blocks in actives/decoys mol2 files via grep-level scan.
  - Count `########## Name:` headers and `Long Name: NO_LONG_NAME` headers.
  - Run the project's `iter_poses()` to count successfully parsed RDKit mols
    (silent sanitization drops are surfaced as a parse-success rate).
  - Classify parsed mol IDs: CHEMBL_NNNN form, ZINC_NNNN form, other, missing.
  - Detect duplicate IDs within the same file.
  - Flag any parsed ID that is literally the string `NO_LONG_NAME`.

Writes a Markdown report to notes/phase3_mol2_audit.md.

Run:
    PYTHONPATH=/home/maurbina/flowr_root \\
        /home/maurbina/.venvs/dc_featurizers/bin/python scripts/audit_dudez_mol2.py
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

_HDBIND = Path("/home/maurbina/hdbind-3D")
if str(_HDBIND) not in sys.path:
    sys.path.insert(0, str(_HDBIND))

from src.data_loading import iter_poses  # noqa: E402
from config import DOCKING_DIR, MOL2_SUBDIR, MOL2_PREFIX  # noqa: E402


CHEMBL_RE = re.compile(r"^CHEMBL\d+$")
ZINC_RE = re.compile(r"^ZINC\d+$|^C\d+$")  # ZINC ids in DUDE-Z sometimes appear as C<digits>


def grep_counts(mol2_path: Path) -> dict:
    """Cheap stats: scan the file linearly without RDKit."""
    n_molecule = 0
    n_name = 0
    n_long_name_NO = 0
    n_long_name_other = 0
    sample_names: list[str] = []
    with open(mol2_path) as f:
        for line in f:
            if line.startswith("@<TRIPOS>MOLECULE"):
                n_molecule += 1
            elif line.startswith("##########"):
                if "Long Name:" in line:
                    val = line.split("Long Name:")[-1].strip()
                    if val == "NO_LONG_NAME":
                        n_long_name_NO += 1
                    else:
                        n_long_name_other += 1
                elif "Name:" in line:  # plain "Name:" only — Long Name handled above
                    n_name += 1
                    if len(sample_names) < 5:
                        sample_names.append(line.split("Name:")[-1].strip())
    return {
        "molecule_blocks": n_molecule,
        "name_headers": n_name,
        "long_name_NO_LONG_NAME": n_long_name_NO,
        "long_name_other": n_long_name_other,
        "first_5_names": sample_names,
    }


def classify_id(s: str) -> str:
    if s is None or s == "":
        return "missing"
    if s == "NO_LONG_NAME":
        return "NO_LONG_NAME_literal"
    if CHEMBL_RE.match(s):
        return "CHEMBL"
    if ZINC_RE.match(s):
        return "ZINC_like"
    return "other"


def parse_with_helper(target: str, kind: str) -> dict:
    """Use the project's iter_poses; track ID classes, dupes, and drop rate."""
    ids: list[str] = []
    for mol_id, mol in iter_poses(target, kind):
        ids.append(mol_id)
    cls = Counter(classify_id(i) for i in ids)
    dupes = Counter(ids)
    dupe_ids = [(i, c) for i, c in dupes.items() if c > 1]
    return {
        "n_parsed": len(ids),
        "id_classes": dict(cls),
        "n_unique_ids": len(set(ids)),
        "duplicate_id_pairs": dupe_ids[:10],  # first 10 only
        "n_duplicate_ids": len(dupe_ids),
        "first_5_ids": ids[:5],
    }


def audit_target(target: str) -> dict:
    out = {"target": target}
    for kind, fname_kind in [("ligand", "ligand"), ("decoy", "decoy")]:
        mol2 = DOCKING_DIR / target / MOL2_SUBDIR / f"{MOL2_PREFIX}_{fname_kind}_poses.mol2"
        if not mol2.exists():
            out[f"{kind}_path"] = str(mol2) + "  [MISSING]"
            continue
        out[f"{kind}_path"] = str(mol2)
        out[f"{kind}_grep"] = grep_counts(mol2)
        out[f"{kind}_parsed"] = parse_with_helper(target, kind)
        g = out[f"{kind}_grep"]
        p = out[f"{kind}_parsed"]
        out[f"{kind}_parse_success_rate"] = (
            p["n_parsed"] / g["molecule_blocks"] if g["molecule_blocks"] else float("nan")
        )
    return out


def render_md(audits: list[dict]) -> str:
    L: list[str] = []
    L.append("# DUDE-Z mol2 input audit — Phase 3 precondition\n")
    L.append("Audit precedes any feature extraction. Confirms that the project's "
             "`iter_poses()` helper bypasses the `NO_LONG_NAME` collision documented "
             "in CLAUDE.md and that pose counts match raw `@<TRIPOS>MOLECULE` block "
             "counts. Source: [scripts/audit_dudez_mol2.py](../scripts/audit_dudez_mol2.py).\n")

    L.append("## Methodology\n")
    L.append("1. **Cheap grep scan** of each mol2 file: count `@<TRIPOS>MOLECULE` "
             "blocks, `########## Name:` headers, and the literal "
             "`########## Long Name: NO_LONG_NAME` lines.\n")
    L.append("2. **Full RDKit parse** via `src.data_loading.iter_poses()`, which "
             "uses the `Name:` (not `Long Name:`) header and sanitizes with "
             "`Chem.SanitizeMol`. Silent sanitization drops surface as parse-success "
             "rate < 1.0.\n")
    L.append("3. **ID classification**: CHEMBL / ZINC-like / NO_LONG_NAME literal / "
             "missing / other. Any non-CHEMBL/ZINC IDs are flagged for inspection.\n")
    L.append("4. **Duplicate ID** scan within each file.\n")

    L.append("## Per-target audit\n")
    for a in audits:
        L.append(f"### {a['target']}\n")
        for kind in ("ligand", "decoy"):
            path_key = f"{kind}_path"
            if path_key not in a:
                continue
            label = "actives (label=1)" if kind == "ligand" else "decoys (label=0)"
            L.append(f"**{label}** — `{a[path_key]}`\n")
            g = a.get(f"{kind}_grep")
            p = a.get(f"{kind}_parsed")
            if g is None or p is None:
                L.append("  _(missing)_\n")
                continue
            rate = a[f"{kind}_parse_success_rate"]
            L.append("| Metric | Value |")
            L.append("|---|---|")
            L.append(f"| `@<TRIPOS>MOLECULE` blocks (grep) | {g['molecule_blocks']} |")
            L.append(f"| `########## Name:` headers (grep) | {g['name_headers']} |")
            L.append(f"| `########## Long Name: NO_LONG_NAME` (grep) | {g['long_name_NO_LONG_NAME']} |")
            L.append(f"| `########## Long Name:` other (grep) | {g['long_name_other']} |")
            L.append(f"| Successfully parsed via `iter_poses` | {p['n_parsed']} |")
            L.append(f"| Parse success rate | {rate:.4f} |")
            L.append(f"| Unique parsed IDs | {p['n_unique_ids']} |")
            L.append(f"| Duplicate ID count | {p['n_duplicate_ids']} |")
            for k, v in p["id_classes"].items():
                L.append(f"| ID class `{k}` | {v} |")
            L.append("")
            L.append(f"First 5 parsed IDs: `{p['first_5_ids']}`")
            L.append("")
            if p["n_duplicate_ids"]:
                L.append(f"⚠️ First duplicate-ID pairs (id, count): "
                         f"`{p['duplicate_id_pairs']}`\n")
            if "NO_LONG_NAME_literal" in p["id_classes"]:
                L.append(f"⚠️ **{p['id_classes']['NO_LONG_NAME_literal']} parsed IDs "
                         "are the literal string `NO_LONG_NAME`.** The Name:/Long Name: "
                         "split in `_parse_mol2` failed for these. Investigate before "
                         "extraction.\n")
            if "other" in p["id_classes"] and p["id_classes"]["other"]:
                L.append(f"⚠️ {p['id_classes']['other']} IDs do not match CHEMBL or "
                         "ZINC patterns — review.\n")

    # Aggregate verdict
    L.append("## Verdict\n")
    blockers = []
    cautions = []
    for a in audits:
        for kind in ("ligand", "decoy"):
            if f"{kind}_grep" not in a:
                continue
            g = a[f"{kind}_grep"]
            p = a[f"{kind}_parsed"]
            rate = a[f"{kind}_parse_success_rate"]
            cls = p["id_classes"]
            tag = f"{a['target']}/{kind}"
            if "NO_LONG_NAME_literal" in cls:
                blockers.append(f"{tag}: parsed ID equals `NO_LONG_NAME` literally — "
                                f"`_parse_mol2` failed to distinguish Name vs Long Name")
            if p["n_duplicate_ids"]:
                cautions.append(f"{tag}: {p['n_duplicate_ids']} duplicate parsed IDs")
            if rate < 0.95:
                cautions.append(f"{tag}: parse-success rate {rate:.4f} (<0.95) — "
                                f"silent RDKit sanitize drops")
            if g["molecule_blocks"] != g["name_headers"]:
                cautions.append(f"{tag}: MOLECULE blocks={g['molecule_blocks']} "
                                f"vs Name headers={g['name_headers']} — header / block "
                                f"count mismatch in upstream file")
    if blockers:
        L.append("**BLOCKERS — do not start extraction:**")
        for b in blockers:
            L.append(f"- {b}")
        L.append("")
    if cautions:
        L.append("**Cautions — note for record:**")
        for c in cautions:
            L.append(f"- {c}")
        L.append("")
    if not blockers and not cautions:
        L.append("All checks passed. Safe to proceed with Phase 3 extraction on "
                 "the audited targets.\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["AA2AR", "ADRB2", "EGFR"])
    ap.add_argument("--output", default=str(_HDBIND / "notes" / "phase3_mol2_audit.md"))
    args = ap.parse_args()

    print(f"Auditing targets: {args.targets}")
    print(f"DOCKING_DIR={DOCKING_DIR}  subdir={MOL2_SUBDIR}  prefix={MOL2_PREFIX}")
    audits = [audit_target(t) for t in args.targets]

    for a in audits:
        print(f"\n=== {a['target']} ===")
        for kind in ("ligand", "decoy"):
            if f"{kind}_grep" not in a:
                continue
            g = a[f"{kind}_grep"]
            p = a[f"{kind}_parsed"]
            rate = a[f"{kind}_parse_success_rate"]
            print(f"  {kind}: grep_blocks={g['molecule_blocks']}  "
                  f"NO_LONG_NAME={g['long_name_NO_LONG_NAME']}  "
                  f"parsed={p['n_parsed']}  rate={rate:.4f}  "
                  f"id_classes={p['id_classes']}")

    Path(args.output).write_text(render_md(audits))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
