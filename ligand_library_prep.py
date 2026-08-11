"""
──────────────────────────────────────────────────────────────────────────────
Usage:
  python lig_library_prep.py [options]

  --input        Input SDF file               (default: DrugBank_2614.sdf)
  --output       Output SDF (PAINS-clean)     (default: rdkit_prepared.sdf)
  --output-pains Output SDF (PAINS flagged)   (default: rdkit_pains_flagged.sdf)
  --failed       Output SDF (failed/removed)  (default: failed_mols.sdf)
  --seed         ETKDGv3 random seed          (default: 42)
  --num-confs    Conformers per molecule      (default: 3)
  --keep-hs      Keep explicit Hs in output   (flag; default: strip Hs)
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import datetime
import hashlib
import logging
import multiprocessing
import os
import sys

import dimorphite_dl

# DataStructs lives at the rdkit top-level, NOT under rdkit.Chem.
# 'from rdkit.Chem import DataStructs' raises AttributeError on all RDKit builds.
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchi, MolToInchiKey


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Publication-grade ligand preparation pipeline for structure-based docking",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--input",        default="DrugBank_2614.sdf",       help="Input SDF file")
parser.add_argument("--output",       default="rdkit_prepared.sdf",      help="Output SDF (docking-ready, PAINS-clean)")
parser.add_argument("--output-pains", default="rdkit_pains_flagged.sdf", help="Output SDF (PAINS/Brenk flagged, secondary set)")
parser.add_argument("--failed",       default="failed_mols.sdf",         help="Output SDF (failed/removed molecules)")
parser.add_argument("--seed",         type=int, default=42,              help="Random seed for ETKDGv3")
parser.add_argument("--num-confs",    type=int, default=3,               help="Conformers per molecule (lowest energy kept)")
parser.add_argument("--keep-hs",      action="store_true",               help="Keep explicit Hs in output (for GNINA/PLANTS)")
args = parser.parse_args()


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("ligand_prep.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

# ── Version strings ───────────────────────────────────────────────────────────
rdkit_version      = rdBase.rdkitVersion
dimorphite_version = getattr(dimorphite_dl, "__version__", "unknown")
if dimorphite_version == "unknown":
    logging.warning(
        "dimorphite-dl version could not be determined (__version__ "
        "attribute missing). Per NOTE-A, pin dimorphite-dl to an exact "
        "version for reproducible protonation-state enumeration order, "
        "and record the resolved version (e.g. `pip show dimorphite-dl`) "
        "in your methods section."
    )
run_timestamp      = datetime.datetime.now().isoformat()

logging.info(f"RDKit {rdkit_version} | dimorphite-dl {dimorphite_version} | seed={args.seed}")
logging.info(f"Input: {args.input} | Output: {args.output}")


# ── Module-level filter thresholds (single definition, referenced everywhere) ─
# Rotatable bond thresholds:
#   ROTBOND_WARN : advisory flag written to SDF property RotBondFlag
#   ROTBOND_HARD : hard rejection limit; molecules above this are discarded
# These are separated so flexible CoA-analogs (13–15 bonds) can be retained
# in the output while being flagged for reviewer attention.
ROTBOND_WARN = 12   # > ROTBOND_WARN → RotBondFlag = "flexible_advisory"
ROTBOND_HARD = 15   # > ROTBOND_HARD → molecule discarded

# Tanimoto similarity guard for tautomer canonicalization (MorganFP r=2).
TAUTOMER_SIMILARITY_THRESHOLD = 0.85


# ── Sanitization flags ────────────────────────────────────────────────────────
# SANITIZE_KEKULIZE omitted: prevents hard failures on unusual heteroaromatics.
# SANITIZE_CLEANUPCHIRALITY omitted: stereo is managed by the explicit audit
# pipeline (steps 6, 7, 8, 13, 16) so we do not want RDKit to silently
# remove unassigned stereo centers during sanitization.
SANITIZE_FLAGS = (
    Chem.SanitizeFlags.SANITIZE_FINDRADICALS     |
    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY   |
    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION   |
    Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
    Chem.SanitizeFlags.SANITIZE_SYMMRINGS        |
    Chem.SanitizeFlags.SANITIZE_PROPERTIES
)

# ── Allowed atomic numbers ────────────────────────────────────────────────────
# H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I — standard organic + heteroatoms.
ALLOWED_ATOMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}


# ── Reproducibility metadata ──────────────────────────────────────────────────
_metadata_lines = [
    "lig_library_prep.py — run metadata",
    "──────────────────────────────────────────────────────────",
    f"Timestamp:              {run_timestamp}",
    f"RDKit version:          {rdkit_version}",
    f"Dimorphite-DL:          {dimorphite_version}",
    f"Python version:         {sys.version.split()[0]}",
    f"Force field:            MMFF94 (UFF fallback)",
    f"ETKDGv3 seed:           {args.seed}",
    f"ETKDGv3 threads:        -1 (all cores; override via OMP_NUM_THREADS)",
    f"OMP_NUM_THREADS:        {os.environ.get('OMP_NUM_THREADS', 'unset')}",
    f"CPU count:              {multiprocessing.cpu_count()}",
    f"Num conformers:         {args.num_confs}",
    f"pH:                     7.4 (±0.5 dimorphite window)",
    f"Protonation order:      enumeration order — NOT pKa-ranked (see NOTE-A)",
    f"InChIKey layer:         Fixed-H (/FixedH; tautomer-sensitive)",
    f"MW filter:              Average MW [150, 700] Da (Descriptors.MolWt)",
    f"RotBond advisory warn:  > {ROTBOND_WARN}",
    f"RotBond hard reject:    > {ROTBOND_HARD}",
    f"Tautomer guard:         Tanimoto >= {TAUTOMER_SIMILARITY_THRESHOLD} (MorganFP r=2, nBits=2048)",
    f"Stereo audit:           Sorted CIP tuples + canonical-rank bond stereo",
    f"HBA definition:         Strict SMARTS-based (CalcNumHBA); Lipinski N+O also reported",
    f"Input:                  {args.input}",
    f"Output (clean):         {args.output}",
    f"Output (PAINS):         {args.output_pains}",
    f"Output (failed):        {args.failed}",
    f"Keep explicit Hs:       {args.keep_hs}",
    "──────────────────────────────────────────────────────────",
]
with open("ligand_prep_metadata.txt", "w") as _mf:
    _mf.write("\n".join(_metadata_lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def safe_sanitize(mol, name: str, label: str) -> bool:
    """
    Run explicit-flag sanitization.

    Returns True on success, False on any RDKit error.
    Uses SANITIZE_FLAGS (module-level) which intentionally omits
    SANITIZE_KEKULIZE and SANITIZE_CLEANUPCHIRALITY.
    """
    try:
        Chem.SanitizeMol(mol, catchErrors=False, sanitizeOps=SANITIZE_FLAGS)
        return True
    except Exception as exc:
        logging.warning(f"{label} ({name}): sanitize failed: {exc}")
        return False


def _write_failed(mol_or_smiles, name: str, reason: str, writer, mol_index: int) -> None:
    """
    Write a sentinel molecule to the failed SDF writer.

    Uses a 1-atom wildcard mol ("*") so that 0-atom SDF V2000 blocks are
    never produced — 0-atom blocks crash PyMOL, MOE, Maestro and most
    standard SDF parsers.

    Parameters
    ----------
    mol_or_smiles : rdkit.Chem.Mol or None
        Original molecule, used only to carry existing properties if not None.
        A "*" sentinel is always written instead to guarantee parsability.
    name : str
        Molecule name for the _Name field.
    reason : str
        Short descriptive failure reason written to the FailReason field.
    writer : Chem.SDWriter
        Open SDWriter for the failed output file.
    mol_index : int
        Loop index i, used only for the warning message on writer error.
    """
    try:
        _fail_mol = Chem.MolFromSmiles("*")
        _fail_mol.SetProp("_Name", name)
        _fail_mol.SetProp("FailReason", reason)
        writer.write(_fail_mol)
    except Exception as exc:
        logging.warning(f"Mol {mol_index} ({name}): failed_writer error ({reason}): {exc}")


def _fixed_h_inchikey(mol) -> str | None:
    """
    Return a tautomer-sensitive deduplication key using fixed-H InChI.

    Computes SHA-256 of the full /FixedH InChI string because RDKit does not
    expose a direct FixedH InChIKey function.

    Falls back to standard InChIKey (logged warning) if /FixedH fails.
    Returns None if both attempts fail.
    """
    try:
        inchi_fh = MolToInchi(mol, options="/FixedH")
        if inchi_fh:
            return hashlib.sha256(inchi_fh.encode()).hexdigest()
    except Exception:
        pass
    # Fallback: standard InChIKey — tautomer-insensitive; log for visibility.
    try:
        key = MolToInchiKey(mol)
        if key:
            logging.warning(
                "_fixed_h_inchikey: /FixedH InChI failed — "
                "falling back to standard InChIKey (tautomer-insensitive)"
            )
            return key
    except Exception:
        pass
    return None


# ── Bond-stereo exclusion set ─────────────────────────────────────────────────
_STEREO_UNSET = {Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY}


def _chiral_centers_tuple(mol) -> tuple:
    """
    Return a sorted tuple of CIP codes for all tetrahedral stereocenters.

    Sorted TUPLE (not set): set comparison silently misses inversions where
    cardinality is preserved but individual assignments change:
      Before: (R, R) → set {'R'} — equal to {'R'} after one center inverts
      After:  (R, S) → set {'R','S'} — only caught when count changes
    Sorted-tuple comparison: ('R','R') != ('R','S') — correctly detected.
    """
    return tuple(sorted(
        cip for _, cip in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    ))


def _bond_stereo_set(mol) -> frozenset:
    """
    Return a frozenset of (canonical_rank_begin, canonical_rank_end, BondStereo)
    for all kekulized double bonds with a defined E/Z assignment.

    Endpoint atoms are identified by their canonical ranks
    (Chem.CanonicalRankAtoms), NOT by bond index.  Bond indices are
    reassigned after SMILES round-trips (dimorphite step 6, tautomer step 7);
    canonical ranks are derived from graph topology and are stable across
    round-trips for the same molecular graph.

    Aromatic bonds (type 1.5 in RDKit) carry no E/Z assignment and are excluded.
    Bonds with STEREONONE or STEREOANY (unspecified) are excluded; only
    defined E/Z assignments are tracked.
    """
    ranks = list(Chem.CanonicalRankAtoms(mol))
    return frozenset(
        (
            ranks[bond.GetBeginAtomIdx()],
            ranks[bond.GetEndAtomIdx()],
            bond.GetStereo(),
        )
        for bond in mol.GetBonds()
        if bond.GetBondTypeAsDouble() == 2.0       # kekulized double bonds only
        and bond.GetStereo() not in _STEREO_UNSET  # defined E/Z only
    )


def _tanimoto_morgan(mol_a, mol_b, radius: int = 2, n_bits: int = 2048) -> float:
    """
    Compute Morgan-fingerprint Tanimoto similarity between two sanitized mols.

    Both mol_a and mol_b MUST be sanitized before calling this function.
    Unsanitized mols (un-kekulized aromatics) silently return zeroed
    fingerprints, making all similarities appear as 0.0.
    """
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


# ── Standardizers (single instantiation — not re-created per molecule) ────────
fragment_chooser = rdMolStandardize.LargestFragmentChooser()
tautomer_canon   = rdMolStandardize.TautomerEnumerator()


# ── PAINS + Brenk catalogs ────────────────────────────────────────────────────
# Individual catalogs are kept for per-source breakdown in statistics.
# A combined catalog is used for the primary PAINS_Brenk property tag.
_params_pains = FilterCatalogParams()
_params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_pains_only      = FilterCatalog(_params_pains)
_n_pains_entries = _pains_only.GetNumEntries()

_params_brenk = FilterCatalogParams()
_params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_brenk_only      = FilterCatalog(_params_brenk)
_n_brenk_entries = _brenk_only.GetNumEntries()

_params_combined = FilterCatalogParams()
_params_combined.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_params_combined.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
pains_catalog = FilterCatalog(_params_combined)

logging.info(
    f"FilterCatalog: PAINS={_n_pains_entries} entries, "
    f"BRENK={_n_brenk_entries} entries, "
    f"combined={pains_catalog.GetNumEntries()} entries"
)
if pains_catalog.GetNumEntries() == 0:
    raise RuntimeError(
        "PAINS/Brenk FilterCatalog is empty — RDKit build may be missing the "
        "BRENK catalog. Aborting to prevent silent pass-through of all molecules."
    )


# ── Statistics counters ───────────────────────────────────────────────────────
# Counters included in total_removed (molecules fully discarded):
#   parse_fail, sanitize_fail_step2, sanitize_fail_step9,
#   fragment_chooser_fail, metal, desalt_fail,
#   duplicate_pre, duplicate_post,
#   protonate_hard_fail,   ← mol is discarded when dimorphite SMILES is unparseable
#   mw, atoms, rotbonds, hbd, hba, extreme_charge, embed_fail
#
# Counters NOT in total_removed (molecule kept, possibly flagged):
#   protonate_no_state_change, protonate_outer_error, protonate_invalid_smiles_kept,
#   stereo_protect_fail, tautomer_stereo_changed, tautomer_similarity_rejected,
#   stereo_changed (post-embed), ff_fallback_to_uff, ff_native_uff, ff_none
stats = {
    "total":                        0,
    # Sanitization — split by stage for unambiguous diagnostics
    "parse_fail":                   0,
    "sanitize_fail_step2":          0,   # step 2: initial parse-time sanitization
    "sanitize_fail_step9":          0,   # step 9: post-standardization sanitization
    # Desalting
    "fragment_chooser_fail":        0,   # step 3: fragment chooser exception (discarded)
    "metal":                        0,
    "desalt_fail":                  0,
    # Deduplication
    "duplicate_pre":                0,
    "duplicate_post":               0,
    # Protonation — semantics:
    #   protonate_hard_fail          → mol discarded (SMILES invalid → parse failed)
    #   protonate_no_state_change    → mol kept unchanged (dimorphite returned empty)
    #   protonate_outer_error        → mol kept unchanged (outer exception in step 6)
    "protonate_hard_fail":          0,
    "protonate_no_state_change":    0,
    "protonate_outer_error":        0,
    # Stereo / tautomer guards — mol kept
    "stereo_protect_fail":          0,
    "tautomer_stereo_changed":      0,
    "tautomer_similarity_rejected": 0,
    # Property filters
    "mw":                           0,
    "atoms":                        0,
    "rotbonds":                     0,
    "hbd":                          0,
    "hba":                          0,
    "extreme_charge":               0,
    # PAINS/Brenk breakdown
    "pains_flagged":                0,
    "pains_only":                   0,
    "brenk_only":                   0,
    "pains_brenk":                  0,
    # 3D embedding
    "embed_fail":                   0,
    "stereo_changed":               0,
    # Force-field counters
    #   ff_fallback_to_uff → MMFF params declared but properties=None → UFF used
    #   ff_native_uff      → mol never had MMFF params; UFF as primary choice
    #   ff_none            → neither FF available; conformers stored unminimized
    "ff_fallback_to_uff":           0,
    "ff_native_uff":                0,
    "ff_none":                      0,
    # Output routing
    "passed_clean":                 0,
    "passed_pains":                 0,
}


# ── I/O setup ─────────────────────────────────────────────────────────────────
seen_pre_inchikeys:  set = set()
seen_post_inchikeys: set = set()

supplier      = Chem.SDMolSupplier(args.input, sanitize=False, removeHs=True)
writer        = Chem.SDWriter(args.output)
pains_writer  = Chem.SDWriter(args.output_pains)
failed_writer = Chem.SDWriter(args.failed)


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────
try:
    for i, mol in enumerate(supplier):
        stats["total"] += 1

        # ── 1. Parse ──────────────────────────────────────────────────────────
        if mol is None:
            logging.warning(f"Mol {i}: parse failed — skipped")
            stats["parse_fail"] += 1
            continue

        # Build a human-readable molecule name from available SDF fields.
        db_id    = mol.GetProp("DRUGBANK_ID").strip()  if mol.HasProp("DRUGBANK_ID")  else ""
        gen_name = mol.GetProp("GENERIC_NAME").strip() if mol.HasProp("GENERIC_NAME") else ""
        raw_name = mol.GetProp("_Name").strip()        if mol.HasProp("_Name")        else ""
        alt_name = (
            mol.GetProp("chembl_id").strip()      if mol.HasProp("chembl_id")     else
            mol.GetProp("zinc_id").strip()        if mol.HasProp("zinc_id")       else
            mol.GetProp("PUBCHEM_CID").strip()    if mol.HasProp("PUBCHEM_CID")   else
            mol.GetProp("CAS").strip()            if mol.HasProp("CAS")           else
            mol.GetProp("Compound_Name").strip()  if mol.HasProp("Compound_Name") else
            mol.GetProp("Name").strip()           if mol.HasProp("Name")          else
            mol.GetProp("ID").strip()             if mol.HasProp("ID")            else ""
        )
        if db_id and gen_name:
            name = f"{db_id}_{gen_name.replace(' ', '_')}"
        elif db_id:
            name = db_id
        elif gen_name:
            name = gen_name.replace(" ", "_")
        elif raw_name and not raw_name.isdigit() and len(raw_name) > 1:
            name = raw_name.replace(" ", "_")
        elif alt_name and not alt_name.isdigit():
            name = alt_name.replace(" ", "_")
        else:
            name = f"MOL_{i + 1:05d}"
        mol.SetProp("_Name", name)

        # ── 2. Initial sanitize ───────────────────────────────────────────────
        if not safe_sanitize(mol, name, f"Mol {i}"):
            stats["sanitize_fail_step2"] += 1
            _write_failed(mol, name, "sanitize_fail_step2", failed_writer, i)
            continue

        # ── 3. Desalt (largest fragment) ──────────────────────────────────────
        # Performed BEFORE the metal/disallowed-atom filter. Many DrugBank/
        # ZINC entries are salts (Na+, K+, Ca2+, Mg2+ counter-ions, etc.) whose
        # counter-ion contains an atom outside ALLOWED_ATOMS even though the
        # parent drug molecule is perfectly valid organic chemistry. Filtering
        # for disallowed atoms BEFORE desalting would discard the entire salt
        # form — including the valid parent — for a counter-ion that is about
        # to be stripped anyway. Desalting first ensures the metal filter (step
        # 4) only judges the molecule that will actually be carried forward.
        _n_frags = len(Chem.GetMolFrags(mol))
        try:
            mol = fragment_chooser.choose(mol)
        except Exception as exc:
            logging.warning(
                f"Mol {i} ({name}): fragment chooser failed "
                f"(kekulization or other error): {exc} — skipped"
            )
            # This is a desalting failure, not a sanitization failure.
            stats["fragment_chooser_fail"] += 1
            _write_failed(mol, name, "fragment_chooser_fail", failed_writer, i)
            continue

        if mol is None or mol.GetNumHeavyAtoms() == 0:
            logging.warning(f"Mol {i} ({name}): empty after desalting — skipped")
            stats["desalt_fail"] += 1
            continue

        if _n_frags > 1:
            logging.info(
                f"Mol {i} ({name}): desalted ({_n_frags} fragments → largest kept)"
            )

        # ── 4. Metal / disallowed atom filter ─────────────────────────────────
        # Applied to the desalted (parent) fragment only — see step 3 rationale.
        if any(a.GetAtomicNum() not in ALLOWED_ATOMS for a in mol.GetAtoms()):
            logging.info(
                f"Mol {i} ({name}): parent fragment contains disallowed atom "
                f"— skipped"
            )
            stats["metal"] += 1
            continue

        # ── 5. Pre-protonation dedup (fixed-H InChIKey) ───────────────────────
        try:
            _pre_mol = Chem.RemoveHs(mol)
            if not safe_sanitize(_pre_mol, name, f"Mol {i} pre-dedup"):
                raise ValueError("sanitize failed on pre-dedup mol")
            pre_key = _fixed_h_inchikey(_pre_mol)
            if not pre_key:
                raise ValueError("empty fixed-H InChIKey at pre-dedup")
            if pre_key in seen_pre_inchikeys:
                logging.info(f"Mol {i} ({name}): pre-prot duplicate (fixed-H) — skipped")
                stats["duplicate_pre"] += 1
                continue
            seen_pre_inchikeys.add(pre_key)
        except Exception as exc:
            logging.warning(
                f"Mol {i} ({name}): pre-prot dedup error — proceeding: {exc}"
            )

        # ── 6. pH 7.4 protonation ─────────────────────────────────────────────
        # NOTE-A: dimorphite-DL SMILES round-trip is an unavoidable API constraint.
        # Bond stereo snapshot uses canonical atom ranks (not bond indices) for
        # stability after the SMILES round-trip.
        # Outer except routes to protonate_outer_error; mol is kept unchanged
        # and is NOT discarded (not counted in total_removed).
        protonation_state = "unknown"
        try:
            _smiles_in = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

            # Stereo snapshot on H-stripped form before protonation.
            _mol_noH_pre             = Chem.RemoveHs(mol)
            _centers_before_prot     = _chiral_centers_tuple(_mol_noH_pre)
            _bond_stereo_before_prot = _bond_stereo_set(_mol_noH_pre)

            # Coerce return value to list[str] immediately (generator exhaustion guard).
            _protonated_list = list(
                dimorphite_dl.protonate_smiles(
                    _smiles_in, ph_min=7.4, ph_max=7.4, precision=0.5
                )
            )

            if _protonated_list:
                if len(_protonated_list) > 1:
                    logging.info(
                        f"Mol {i} ({name}): {len(_protonated_list)} protonation "
                        f"states returned — using first (enumeration order; "
                        f"NOT pKa-ranked; see NOTE-A)"
                    )
                mol_p = Chem.MolFromSmiles(_protonated_list[0])

                if mol_p is not None:
                    # mol_p is valid; proceed with stereo reconciliation.
                    try:
                        if not safe_sanitize(mol_p, name, f"Mol {i} post-prot"):
                            raise ValueError("post-prot sanitize failed")
                        Chem.AssignStereochemistry(mol_p, cleanIt=True, force=True)

                        # Stereo comparison on H-stripped form (prevents false
                        # alerts from explicit-H vs implicit-H differences).
                        _mol_p_noH              = Chem.RemoveHs(mol_p)
                        _centers_after_prot     = _chiral_centers_tuple(_mol_p_noH)
                        _bond_stereo_after_prot = _bond_stereo_set(_mol_p_noH)

                        _center_chg = _centers_before_prot != _centers_after_prot
                        _bond_chg   = _bond_stereo_before_prot != _bond_stereo_after_prot

                        if _center_chg or _bond_chg:
                            logging.warning(
                                f"Mol {i} ({name}): stereo altered during protonation "
                                f"(centers={_center_chg}, bond_stereo={_bond_chg}) "
                                f"— keeping pre-prot structure"
                            )
                            stats["stereo_protect_fail"] += 1
                            protonation_state = "rejected"
                            # mol is unchanged; name already set
                        else:
                            mol_p.SetProp("_Name", name)
                            mol               = mol_p
                            protonation_state = "applied"

                    except Exception as exc:
                        logging.warning(
                            f"Mol {i} ({name}): post-prot processing failed: "
                            f"{exc} — keeping pre-prot structure"
                        )
                        protonation_state = "post_prot_error_kept"

                else:
                    # dimorphite returned non-empty but SMILES is unparseable.
                    # The original mol cannot be trusted to represent pH 7.4 state
                    # because dimorphite produced output but it is invalid.
                    # Discard and record in total_removed.
                    logging.warning(
                        f"Mol {i} ({name}): protonated SMILES produced an invalid "
                        f"mol — discarded (protonate_hard_fail)"
                    )
                    stats["protonate_hard_fail"] += 1
                    _write_failed(mol, name, "protonate_hard_fail_invalid_smiles", failed_writer, i)
                    continue

            else:
                # dimorphite returned empty list — no pH state available.
                # Keep original structure; not a discard (not in total_removed).
                logging.warning(
                    f"Mol {i} ({name}): dimorphite returned empty list "
                    f"— keeping original (state=kept_original)"
                )
                stats["protonate_no_state_change"] += 1
                protonation_state = "kept_original"

        except Exception as exc:
            # Outer except: reached if, e.g., MolToSmiles() raises on an
            # edge-case aromatic system. The mol object is still valid and is
            # passed downstream unchanged. NOT a hard failure (not discarded,
            # not in total_removed).
            logging.warning(
                f"Mol {i} ({name}): protonation outer error: {exc} "
                f"— state=unknown, mol kept unchanged"
            )
            stats["protonate_outer_error"] += 1
            protonation_state = "unknown"

        # ── 7. Tautomer canonicalization ──────────────────────────────────────
        # Candidate is sanitized before Tanimoto comparison to prevent silent
        # zero fingerprints from un-kekulized aromatic systems.
        # Tanimoto guard (>= TAUTOMER_SIMILARITY_THRESHOLD) prevents
        # pharmacophore drift.  Sorted-tuple stereo audit rejects tautomers
        # that silently alter tetrahedral or E/Z assignments.
        #
        # SMILES comparison for logging uses H-stripped canonical SMILES for
        # both before and after to avoid spurious "tautomer form updated"
        # messages caused by H-representation differences after the protonation
        # round-trip.
        try:
            _mol_noH_tau         = Chem.RemoveHs(mol)
            _centers_before_taut = _chiral_centers_tuple(_mol_noH_tau)
            _bond_before_taut    = _bond_stereo_set(_mol_noH_tau)
            # SMILES snapshot on H-stripped form for apples-to-apples comparison.
            _smi_before = Chem.MolToSmiles(_mol_noH_tau, isomericSmiles=True, canonical=True)

            mol_candidate = tautomer_canon.Canonicalize(mol)

            # Sanitize candidate before fingerprinting.
            _cand_noH = Chem.RemoveHs(mol_candidate)
            if not safe_sanitize(_cand_noH, name, f"Mol {i} taut-candidate"):
                logging.warning(
                    f"Mol {i} ({name}): tautomer candidate failed sanitize "
                    f"— keeping pre-tautomer structure"
                )
                stats["tautomer_similarity_rejected"] += 1
            else:
                _tanimoto = _tanimoto_morgan(_mol_noH_tau, _cand_noH)
                if _tanimoto < TAUTOMER_SIMILARITY_THRESHOLD:
                    logging.warning(
                        f"Mol {i} ({name}): tautomer Tanimoto={_tanimoto:.3f} "
                        f"< {TAUTOMER_SIMILARITY_THRESHOLD} threshold "
                        f"— keeping pre-tautomer structure (pharmacophore drift guard)"
                    )
                    stats["tautomer_similarity_rejected"] += 1
                else:
                    _centers_after_taut = _chiral_centers_tuple(_cand_noH)
                    _bond_after_taut    = _bond_stereo_set(_cand_noH)
                    _c_chg = _centers_before_taut != _centers_after_taut
                    _b_chg = _bond_before_taut    != _bond_after_taut

                    if _c_chg or _b_chg:
                        logging.warning(
                            f"Mol {i} ({name}): tautomer altered stereo "
                            f"(centers={_c_chg}, bond={_b_chg}) "
                            f"— keeping pre-tautomer structure"
                        )
                        stats["tautomer_stereo_changed"] += 1
                    else:
                        # Both SMILES are H-stripped canonical — comparison is reliable.
                        _smi_after = Chem.MolToSmiles(
                            _cand_noH, isomericSmiles=True, canonical=True
                        )
                        if _smi_before != _smi_after:
                            logging.info(f"Mol {i} ({name}): tautomer form updated")
                        mol = mol_candidate

        except Exception as exc:
            logging.warning(
                f"Mol {i} ({name}): tautomer canonicalization error: {exc} "
                f"— keeping current structure"
            )

        # ── 8. Assign stereochemistry ─────────────────────────────────────────
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # ── 9. Post-standardization sanitize ─────────────────────────────────
        if not safe_sanitize(mol, name, f"Mol {i} post-std"):
            stats["sanitize_fail_step9"] += 1
            _write_failed(mol, name, "sanitize_fail_step9", failed_writer, i)
            continue

        # ── 10. Post-tautomer dedup (fixed-H InChIKey) ────────────────────────
        try:
            _post_mol = Chem.RemoveHs(mol)
            if not safe_sanitize(_post_mol, name, f"Mol {i} post-dedup"):
                raise ValueError("sanitize failed on post-dedup mol")
            post_key = _fixed_h_inchikey(_post_mol)
            if not post_key:
                raise ValueError("empty fixed-H InChIKey at post-dedup")
            if post_key in seen_post_inchikeys:
                logging.info(
                    f"Mol {i} ({name}): post-tautomer duplicate (fixed-H) — skipped"
                )
                stats["duplicate_post"] += 1
                continue
            seen_post_inchikeys.add(post_key)
        except Exception as exc:
            logging.warning(
                f"Mol {i} ({name}): post-tautomer dedup error — proceeding: {exc}"
            )

        # ── 11. Property filters ──────────────────────────────────────────────
        # avg MW (MolWt) is used for the [150, 700] filter (Lipinski average MW).
        # ExactMolWt is computed and attached to the output as an annotation only.
        avg_mw    = Descriptors.MolWt(mol)
        exact_mw  = Descriptors.ExactMolWt(mol)
        num_atoms = mol.GetNumHeavyAtoms()
        rot_bonds   = Descriptors.NumRotatableBonds(mol)
        # HBA: stricter SMARTS-based acceptor count (rdMolDescriptors.CalcNumHBA),
        # which excludes amide/aniline/pyrrole-type nitrogens and other N/O atoms
        # that are not real H-bond acceptors in practice (Eddington/MDDR-style
        # definition). This avoids over-filtering amide-rich molecules that a
        # plain Lipinski N+O count (Descriptors.NumHAcceptors) would penalize.
        # The legacy Lipinski N+O count is still computed and reported as
        # HBA_LipinskiNO for cross-reference / comparison with older datasets.
        hba             = rdMolDescriptors.CalcNumHBA(mol)
        hba_lipinski_no = Descriptors.NumHAcceptors(mol)   # Lipinski N+O count (reporting only)
        hbd       = Descriptors.NumHDonors(mol)
        logp      = Descriptors.MolLogP(mol)
        tpsa      = Descriptors.TPSA(mol)
        charge    = Chem.GetFormalCharge(mol)

        if not (150 <= avg_mw <= 700):
            logging.info(
                f"Mol {i} ({name}): avg MW={avg_mw:.1f} outside [150, 700] — skipped"
            )
            stats["mw"] += 1
            continue

        if num_atoms > 70:
            logging.info(
                f"Mol {i} ({name}): heavy atoms={num_atoms} > 70 — skipped"
            )
            stats["atoms"] += 1
            continue

        # Rotatable bond filter: two-tier (warn / hard-reject).
        if rot_bonds > ROTBOND_HARD:
            logging.info(
                f"Mol {i} ({name}): rot_bonds={rot_bonds} > {ROTBOND_HARD} "
                f"(hard limit) — skipped"
            )
            stats["rotbonds"] += 1
            continue

        rotbond_flag = "flexible_advisory" if rot_bonds > ROTBOND_WARN else "OK"
        if rot_bonds > ROTBOND_WARN:
            logging.info(
                f"Mol {i} ({name}): rot_bonds={rot_bonds} in advisory range "
                f"[{ROTBOND_WARN + 1}–{ROTBOND_HARD}] — kept with advisory flag"
            )

        if hbd > 5:
            logging.info(
                f"Mol {i} ({name}): HBD={hbd} > 5 (Lipinski) — skipped"
            )
            stats["hbd"] += 1
            continue

        if hba > 10:
            logging.info(
                f"Mol {i} ({name}): HBA={hba} (strict SMARTS def.) "
                f"> 10 — skipped (Lipinski N+O count was {hba_lipinski_no})"
            )
            stats["hba"] += 1
            continue

        # Conformer-count advisory (NOTE-E): tiered recommendation for flexible mols.
        _confs_needed = 50 if rot_bonds > 9 else (20 if rot_bonds > 6 else 10)
        if args.num_confs < _confs_needed:
            logging.warning(
                f"Mol {i} ({name}): rot_bonds={rot_bonds} — "
                f"recommended >= {_confs_needed} conformers, "
                f"only {args.num_confs} requested. See NOTE-E."
            )

        # Formal charge filter.
        if abs(charge) > 2:
            logging.info(
                f"Mol {i} ({name}): FormalCharge={charge} (|q|>2) "
                f"— extreme ionization state, skipped"
            )
            stats["extreme_charge"] += 1
            _write_failed(mol, name, f"extreme_charge_{charge}", failed_writer, i)
            continue

        if abs(charge) in (1, 2):
            logging.warning(
                f"Mol {i} ({name}): FormalCharge={charge} — non-zero charge; "
                f"verify docking scoring function handles this correctly"
            )

        # ── 12. PAINS + Brenk filter (breakdown by catalog) ──────────────────
        # Molecules are not discarded — they are routed to the secondary output
        # file so reviewers can inspect and manually promote if warranted.
        _hit_pains   = _pains_only.HasMatch(mol)
        _hit_brenk   = _brenk_only.HasMatch(mol)
        is_pains     = _hit_pains or _hit_brenk
        pains_flag   = "None"
        pains_source = "None"

        if is_pains:
            _entry = pains_catalog.GetFirstMatch(mol)
            pains_flag = _entry.GetDescription() if _entry is not None else "unknown"
            if _hit_pains and _hit_brenk:
                stats["pains_brenk"] += 1
                pains_source = "PAINS+Brenk"
            elif _hit_pains:
                stats["pains_only"] += 1
                pains_source = "PAINS"
            else:
                stats["brenk_only"] += 1
                pains_source = "Brenk"
            logging.info(
                f"Mol {i} ({name}): {pains_source} hit — {pains_flag} "
                f"(routed to secondary set)"
            )
            stats["pains_flagged"] += 1

        # ── 13. Chirality snapshot BEFORE embedding ───────────────────────────
        # Snapshot on H-stripped form for consistency with the post-embedding
        # audit (step 16), preventing false alerts from explicit-H differences.
        _snap_noH          = Chem.RemoveHs(mol)
        chiral_before      = _chiral_centers_tuple(_snap_noH)
        bond_stereo_before = _bond_stereo_set(_snap_noH)

        # ── 14. Add Hs + ETKDGv3 multi-conformer embedding ───────────────────
        mol_h = Chem.AddHs(Chem.RemoveHs(mol))

        _etkdg                  = AllChem.ETKDGv3()
        _etkdg.randomSeed       = args.seed
        _etkdg.numThreads       = -1    # use all available cores; RDKit >= 2020.09
                                        # silently ignored on older builds (no crash)
        _etkdg.enforceChirality = True  # stereo finalised after step 8

        conf_ids = AllChem.EmbedMultipleConfs(
            mol_h, numConfs=args.num_confs, params=_etkdg
        )

        if len(conf_ids) == 0:
            logging.warning(
                f"Mol {i} ({name}): all {args.num_confs} ETKDGv3 embeddings failed"
            )
            stats["embed_fail"] += 1
            # Write H-stripped mol for H-consistency with all other failed entries.
            _write_failed(Chem.RemoveHs(mol_h), name, "embed_fail", failed_writer, i)
            continue

        if len(conf_ids) < args.num_confs:
            logging.info(
                f"Mol {i} ({name}): partial embedding "
                f"({len(conf_ids)}/{args.num_confs} conformers generated)"
            )

        # ── 15. Force-field minimization ──────────────────────────────────────
        # ff_mode is resolved once before the conformer loop:
        #   'MMFF94' → MMFF params available and properties object is valid
        #   'UFF'    → either MMFF params absent (native UFF) or
        #               MMFFGetMoleculeProperties returned None (fallback UFF)
        #   'NONE'   → neither FF available; conformers stored unminimized
        # mmff_fell_back_to_uff distinguishes native UFF from MMFF→UFF fallback
        # so that ff_native_uff is not inflated by genuine fallback events.
        best_conf_id          = conf_ids[0]
        best_energy           = float("inf")
        ff_type               = "none"
        mol_mp                = None
        ff_mode               = "NONE"
        mmff_fell_back_to_uff = False

        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            mol_mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
            if mol_mp is not None:
                ff_mode = "MMFF94"
            else:
                # Edge case: params declared but properties object is None.
                # Seen on molecules with unusual valence in some RDKit builds.
                logging.warning(
                    f"Mol {i} ({name}): MMFFGetMoleculeProperties()=None despite "
                    f"MMFFHasAllMoleculeParams()=True — falling back to UFF"
                )
                mmff_fell_back_to_uff = True
                ff_mode = "UFF" if AllChem.UFFHasAllMoleculeParams(mol_h) else "NONE"
        elif AllChem.UFFHasAllMoleculeParams(mol_h):
            ff_mode = "UFF"   # mmff_fell_back_to_uff remains False → native UFF

        # Increment FF counters exactly once after ff_mode is resolved.
        if ff_mode == "MMFF94":
            pass  # counted at output time via ff_type
        elif ff_mode == "UFF":
            if mmff_fell_back_to_uff:
                stats["ff_fallback_to_uff"] += 1
            else:
                stats["ff_native_uff"] += 1
        else:  # NONE
            stats["ff_none"] += 1
            logging.warning(
                f"Mol {i} ({name}): no MMFF94 or UFF params available "
                f"— conformers stored unminimized"
            )

        _use_mmff = (ff_mode == "MMFF94")
        _use_uff  = (ff_mode == "UFF")

        for conf_id in conf_ids:
            _energy        = None
            _conf_ff_label = "none"

            if _use_mmff:
                try:
                    if mol_mp is None:
                        # Defensive guard: pre-flight check should prevent this.
                        logging.warning(
                            f"Mol {i} ({name}) conf {conf_id}: "
                            f"mol_mp=None inside loop (defensive guard) — skipping"
                        )
                    else:
                        _ff = AllChem.MMFFGetMoleculeForceField(
                            mol_h, mol_mp, confId=conf_id
                        )
                        if _ff is not None:
                            if _ff.Minimize(maxIts=500) == 1:
                                logging.info(
                                    f"Mol {i} ({name}) conf {conf_id}: "
                                    f"MMFF94 did not converge in 500 iterations"
                                )
                            _energy        = _ff.CalcEnergy()
                            _conf_ff_label = "MMFF94"
                            ff_type        = "MMFF94"
                except Exception as exc:
                    logging.warning(
                        f"Mol {i} ({name}) conf {conf_id}: MMFF94 error: {exc}"
                    )

            elif _use_uff:
                try:
                    _ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=conf_id)
                    if _ff is not None:
                        if _ff.Minimize(maxIts=500) == 1:
                            logging.info(
                                f"Mol {i} ({name}) conf {conf_id}: "
                                f"UFF did not converge in 500 iterations"
                            )
                        _energy        = _ff.CalcEnergy()
                        _conf_ff_label = "UFF"
                        ff_type        = "UFF"
                except Exception as exc:
                    logging.warning(
                        f"Mol {i} ({name}) conf {conf_id}: UFF error: {exc}"
                    )

            logging.info(
                f"Mol {i} ({name}) conf {conf_id}: "
                f"FF={_conf_ff_label} energy={_energy}"
            )
            if _energy is not None and _energy < best_energy:
                best_energy  = _energy
                best_conf_id = conf_id

        if ff_mode == "UFF":
            logging.info(
                f"Mol {i} ({name}): UFF used "
                f"(fallback_from_MMFF={mmff_fell_back_to_uff})"
            )

        # Build mol_best with only the lowest-energy conformer.
        # RDKit C++ objects do not support deepcopy reliably across builds.
        # RWMol constructor + AddConformer is the version-stable idiom.
        mol_best = Chem.RWMol(mol_h)
        mol_best.RemoveAllConformers()
        mol_best.AddConformer(mol_h.GetConformer(best_conf_id), assignId=True)
        mol_best = mol_best.GetMol()

        # ── 16. Chirality audit AFTER embedding ───────────────────────────────
        _audit_noH        = Chem.RemoveHs(mol_best)
        chiral_after      = _chiral_centers_tuple(_audit_noH)
        bond_stereo_after = _bond_stereo_set(_audit_noH)

        _c_chg_emb = chiral_before      != chiral_after
        _b_chg_emb = bond_stereo_before != bond_stereo_after
        stereo_ok  = "Yes"

        if _c_chg_emb or _b_chg_emb:
            logging.warning(
                f"Mol {i} ({name}): stereo changed after embedding — "
                f"centers={_c_chg_emb}, bond={_b_chg_emb} "
                f"| before={chiral_before} after={chiral_after}"
            )
            stats["stereo_changed"] += 1
            stereo_ok = "No"

        # ── 17. Strip Hs ──────────────────────────────────────────────────────
        mol_out = mol_best if args.keep_hs else Chem.RemoveHs(mol_best)

        # ── 18. Attach all properties + write ─────────────────────────────────
        mol_out.SetProp("_Name",            name)
        mol_out.SetProp("AverageMW",        f"{avg_mw:.4f}")
        mol_out.SetProp("ExactMW",          f"{exact_mw:.4f}")
        mol_out.SetProp("RotBonds",         str(rot_bonds))
        mol_out.SetProp("HBA",              str(hba))
        mol_out.SetProp("HBA_LipinskiNO",   str(hba_lipinski_no))
        mol_out.SetProp("HBD",              str(hbd))
        mol_out.SetProp("LogP",             f"{logp:.2f}")
        mol_out.SetProp("TPSA",             f"{tpsa:.2f}")
        mol_out.SetProp("FormalCharge",     str(charge))
        mol_out.SetProp("DRUGBANK_ID",      db_id    if db_id    else "unknown")
        mol_out.SetProp("GENERIC_NAME",     gen_name if gen_name else "unknown")
        mol_out.SetProp("HeavyAtoms",       str(num_atoms))
        mol_out.SetProp("ChiralCenters",    str(len(chiral_before)))
        mol_out.SetProp("StereoPreserved",  stereo_ok)
        mol_out.SetProp("ProtonationState", protonation_state)
        mol_out.SetProp("PAINS_Brenk",      pains_flag)
        mol_out.SetProp("PAINS_Source",     pains_source if is_pains else "None")
        mol_out.SetProp("RotBondFlag",      rotbond_flag)
        mol_out.SetProp("ConfsGenerated",   str(len(conf_ids)))
        mol_out.SetProp("ForceField",       ff_type)
        mol_out.SetProp("RDKitVersion",     rdkit_version)
        # Relative FF potential energy (kcal/mol) — NOT absolute enthalpy.
        if best_energy < float("inf"):
            mol_out.SetProp("BestConfRelEnergy_FF_kcalmol", f"{best_energy:.4f}")

        if is_pains:
            pains_writer.write(mol_out)
            stats["passed_pains"] += 1
        else:
            writer.write(mol_out)
            stats["passed_clean"] += 1

        _n_written = stats["passed_clean"] + stats["passed_pains"]
        if _n_written > 0 and _n_written % 50 == 0:
            logging.info(f"── Progress: {_n_written} molecules written ──")

finally:
    writer.close()
    pains_writer.close()
    failed_writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
total_passed = stats["passed_clean"] + stats["passed_pains"]

# total_removed: molecules fully discarded (absent from all output files).
# Counters NOT included (molecule kept, possibly flagged):
#   protonate_no_state_change, protonate_outer_error,
#   stereo_protect_fail, tautomer_stereo_changed, tautomer_similarity_rejected,
#   stereo_changed (post-embed), ff_fallback_to_uff, ff_native_uff, ff_none
_total_sanitize = stats["sanitize_fail_step2"] + stats["sanitize_fail_step9"]
total_removed = (
    stats["parse_fail"]             +
    _total_sanitize                 +
    stats["fragment_chooser_fail"]  +
    stats["metal"]                  +
    stats["desalt_fail"]            +
    stats["duplicate_pre"]          +
    stats["duplicate_post"]         +
    stats["protonate_hard_fail"]    +
    stats["mw"]                     +
    stats["atoms"]                  +
    stats["rotbonds"]               +
    stats["hbd"]                    +
    stats["hba"]                    +
    stats["extreme_charge"]         +
    stats["embed_fail"]
)

summary_lines = [
    "",
    "=" * 66,
    "                LIGAND PREPARATION SUMMARY",
    "=" * 66,
    f"  RDKit:                              {rdkit_version}",
    f"  Dimorphite-DL:                      {dimorphite_version}",
    f"  Timestamp:                          {run_timestamp}",
    f"  {'─' * 62}",
    f"  Total input:                        {stats['total']}",
    "",
    f"  ── Failures (counted in TOTAL REMOVED) ──────────────────────",
    f"  Parse failures:                     {stats['parse_fail']}",
    f"  Sanitize fail — step 2 (initial):   {stats['sanitize_fail_step2']}",
    f"  Sanitize fail — step 9 (post-std):  {stats['sanitize_fail_step9']}",
    f"  Fragment chooser fail (desalt):     {stats['fragment_chooser_fail']}",
    f"  Metal / disallowed atom:            {stats['metal']}",
    f"  Empty after desalting:              {stats['desalt_fail']}",
    "",
    f"  ── Deduplication (counted in TOTAL REMOVED) ─────────────────",
    f"  Pre-prot duplicates (fixed-H key):  {stats['duplicate_pre']}",
    f"  Post-tautomer dups (fixed-H key):   {stats['duplicate_post']}",
    "",
    f"  ── Protonation ──────────────────────────────────────────────",
    f"  Hard failures — discarded:          {stats['protonate_hard_fail']}",
    f"  No pH change / empty — kept:        {stats['protonate_no_state_change']}",
    f"  Outer error — kept:                 {stats['protonate_outer_error']}",
    f"  Stereo protected — kept:            {stats['stereo_protect_fail']}",
    "",
    f"  ── Tautomer guard ───────────────────────────────────────────",
    f"  Stereo changed — kept:              {stats['tautomer_stereo_changed']}",
    f"  Similarity rejected — kept:         {stats['tautomer_similarity_rejected']}",
    "",
    f"  ── Property filters (counted in TOTAL REMOVED) ──────────────",
    f"  Avg MW outside [150, 700]:          {stats['mw']}",
    f"  Heavy atoms > 70:                   {stats['atoms']}",
    f"  RotBonds > {ROTBOND_HARD} (hard):              {stats['rotbonds']}",
    f"  HBD > 5 (Lipinski):                {stats['hbd']}",
    f"  HBA > 10 (strict SMARTS, NOTE-D):  {stats['hba']}",
    f"  Extreme charge |q| > 2:            {stats['extreme_charge']}",
    "",
    f"  ── 3D embedding ─────────────────────────────────────────────",
    f"  Embedding failures — discarded:     {stats['embed_fail']}",
    f"  MMFF→UFF fallback — minimized:      {stats['ff_fallback_to_uff']}",
    f"  Native UFF (no MMFF params):        {stats['ff_native_uff']}",
    f"  No FF — unminimized:                {stats['ff_none']}",
    f"  Stereo changed post-embed — kept:   {stats['stereo_changed']}",
    "",
    f"  ── PAINS / Brenk ────────────────────────────────────────────",
    f"  PAINS only:                         {stats['pains_only']}",
    f"  Brenk only:                         {stats['brenk_only']}",
    f"  PAINS + Brenk:                      {stats['pains_brenk']}",
    f"  Total flagged (secondary set):      {stats['pains_flagged']}",
    "",
    f"  {'─' * 62}",
    f"  PASSED — clean:         {stats['passed_clean']:>6}   →  {args.output}",
    f"  PASSED — PAINS/Brenk:  {stats['passed_pains']:>6}   →  {args.output_pains}",
    f"  TOTAL PASSED:           {total_passed:>6}",
    f"  TOTAL REMOVED:          {total_removed:>6}",
    f"  {'─' * 62}",
    f"  Counters NOT in TOTAL REMOVED (mols kept, possibly flagged):",
    f"    protonate_no_state_change    protonate_outer_error",
    f"    stereo_protect_fail          tautomer_stereo_changed",
    f"    tautomer_similarity_rejected stereo_changed (post-embed)",
    f"    ff_fallback_to_uff           ff_native_uff   ff_none",
    "=" * 66,
]

for line in summary_lines:
    print(line)

with open("ligand_prep_metadata.txt", "a") as _mf:
    _mf.write("\n" + "\n".join(summary_lines) + "\n")
