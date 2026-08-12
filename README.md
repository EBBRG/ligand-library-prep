# Ligand Library Prep

### RDKit Pipeline for Docking-Ready Ligand Preparation
<div align="justify">
Ligand Library Prep is a Python pipeline that transforms raw ligand libraries (SDF) into **docking-ready, PAINS-cleaned, charge-assigned, 3D-validated** ligand collections. It implements the standard cheminformatics preparation cascade required before structure-based virtual screening — from curation and standardisation through PAINS/Brenk filtering, protonation-state enumeration, conformer generation (ETKDGv3), and duplicate/artefact control.

Developed and maintained by the **Evo Biology and Bioinformatics Research Group (EBBRG)**, University of Agriculture Faisalabad.

---




## Table of Contents

- [Features](#features)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Inputs & Outputs](#inputs--outputs)
- [Reproducibility](#reproducibility)
- [Dependencies](#dependencies)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Curation & standardisation** — salt-stripping, neutralisation, tautomer enumeration, canonicalisation via RDKit `MolStandardize`
- **PAINS + Brenk filtering** — combined structural alert catalogs; flagged molecules are written to a separate set, not silently discarded
- **Protonation-state enumeration** — reproducible enumeration (fixed seed) via `dimorphite_dl`, tuned for screening pH conditions
- **3D conformer generation** — ETKDGv3 (RDKit `AllChem`) with configurable conformer count and random seed
- **Charge assignment** — Gasteiger charges and hydrogen handling for downstream docking tools (AutoDock Vina / AutoDock4)
- **Duplicate & artefact control** — InChIKey-based deduplication, chirality/stereo preservation checks, Tanimoto (Morgan, r=2, 2048-bit) redundancy analysis
- **Fault tolerance** — per-molecule error isolation; failed molecules are logged with reasons and written to a dedicated output for audit
- **Scalable** — multiprocessing for library-scale runs (threads configurable via `OMP_NUM_THREADS`)
- **Fully offline** — no web services; everything runs locally

---

## Workflow

```
Raw SDF library (e.g. DrugBank)
            │
            ▼
  1. Sanitisation & curation ──► 2. PAINS/Brenk filtering
            │                                │
            ▼                                ▼
  3. Protonation-state enumeration ──► 4. 3D conformer generation (ETKDGv3)
            │
            ▼
  5. Charge assignment + H handling
            │
            ▼
  Output SDF sets (PAINS-clean · flagged · failed)
```

---

## Installation

### Requirements

- Python ≥ 3.10
- RDKit ≥ 2023.03
- dimorphite_dl

```bash
git clone https://github.com/EBBRG/ligand-library-prep.git
cd ligand-library-prep
pip install -r requirements.txt
```

---

## Usage

```bash
python ligand_library_prep.py \
    --input DrugBank_2614.sdf \
    --output rdkit_prepared.sdf \
    --output-pains rdkit_pains_flagged.sdf \
    --failed failed_mols.sdf \
    --seed 42 \
    --num-confs 3
```

### Key options

| Option | Default | Description |
|---|---|---|
| `--input` | `DrugBank_2614.sdf` | Input SDF library |
| `--output` | `rdkit_prepared.sdf` | Docking-ready, PAINS-clean output |
| `--output-pains` | `rdkit_pains_flagged.sdf` | Molecules flagged by PAINS/Brenk |
| `--failed` | `failed_mols.sdf` | Molecules that failed preparation (with reason) |
| `--seed` | `42` | ETKDGv3 random seed (reproducibility) |
| `--num-confs` | `3` | Conformers per molecule |
| `--keep-hs` | off | Keep explicit hydrogens in output |

---

## Inputs & Outputs

**Input:** a standard SDF file with 2D or 3D molecules (e.g. DrugBank, ChEMBL, ZINC subset).

**Outputs:**

| File | Contents |
|---|---|
| `<output>` | PAINS-clean, protonated, 3D, charge-assigned molecules (docking-ready) |
| `<output-pains>` | Molecules flagged by PAINS/Brenk alerts (kept for audit) |
| `<failed>` | Molecules removed during preparation, each tagged with a failure reason |

---

## Reproducibility

- Fixed random seed (default `42`) for ETKDGv3 conformer generation
- Deterministic protonation-state enumeration order
- Full per-run parameter report printed at start-up
- Every molecule decision is auditable (kept / flagged / failed + reason)

---

## Dependencies

| Package | Purpose |
|---|---|
| [RDKit](https://www.rdkit.org/) | Molecular handling, conformers, PAINS/Brenk catalogs, descriptors |
| [dimorphite-dl](https://github.com/ypcrts/DimorphiteDL) | Protonation-state enumeration |
| Python stdlib | Multiprocessing, hashing, logging |

---

## Repository Structure

```
ligand-library-prep/
│
├── ligand_library_prep.py   # Main pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Citation

If you use this pipeline in your research, please cite the EBBRG group and link to this repository:

> Evo Biology and Bioinformatics Research Group (EBBRG). *Ligand Library Prep: Publication-Grade RDKit Pipeline for Docking-Ready Ligand Preparation.* University of Agriculture Faisalabad. https://github.com/EBBRG/ligand-library-prep

---

## License

Released under the **MIT License**. See `LICENSE`.

---

## Contact

**Evo Biology and Bioinformatics Research Group (EBBRG)**
University of Agriculture Faisalabad, Pakistan

For questions, bug reports, or feature requests, please use the GitHub issue tracker.

</div>
