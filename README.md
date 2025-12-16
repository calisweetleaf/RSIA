# RSIA Provenance Package

## Provenance Overview

# Zenodo DOI: (10.5281/zenodo.17638059) <https://doi.org/10.5281/zenodo.17638059>

This drop contains the runnable symbolic components described in the paper *Recursive Symbolic Identity Architecture (RSIA)*. It is intended to provide a reproducible, hash-signed snapshot of the FBS tokenizer, the enhanced RSGT engine, the RSIA↔RCF bridge, and the NumPy-only RCF core. The neural URFT/base-tensor orchestration remains private and is referenced only conceptually in the manuscript. The integrated RSIA demo runs today, but the orchestrator still needs tuning; until that work lands, the sacred FBS pipeline, RCF core, and RSGT engine are the components we encourage reviewers to exercise independently.

## Repository Layout

- `docs/Recursive Symbolic Identity Architechture.md` — Source manuscript with math foundations and code overviews.
- `docs/RSIA_final.pdf` — XeLaTeX/Pandoc build of the paper.
- `rsgt_snippet.py` — Enhanced RSGT engine with RSIA↔RCF integration.
- `rsia_rcf_bridge.py` — Adapter that feeds RSGT output into the RCF core.(still to be added)
- `rcf_core.py` — NumPy-only reference implementation of the recursive categorical framework.
- `sacred_fbs_tokenizer.py`, `test_sacred_fbs.py` — Post-token FBS pipeline and validation suite.
- `hash-index.ps1` — Provenance tool that writes the `SHA256SUMS.txt` manifest (signed hashes/Timestamp proofs optional).
- `SHA256SUMS.txt` — Hash manifest for all tracked files.

## Running the Demo Components

1. **Sacred FBS tokenizer validation**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python test_sacred_fbs.py
   ```

   This writes metrics and `sacred_fbs_validation.png` for each test.

2. **RSGT engine with RSIA↔RCF bridge**

   ```powershell
   python rsgt_snippet.py
   ```

   The script prints all 10 RSGT tests and ends with the bridge grounding summary (the same state is fed into `rsia_rcf_bridge.py`).

3. **RCF core (standalone)**

   ```powershell
   python rcf_core.py
   ```

   This exercises the NumPy-only categorical engine described in the paper.



## Licensing

This repository is now fully open for research collaboration under CC BY-NC-SA 4.0.


## Citation

Please cite the manuscript as:

> Rowell, C. T. (2025). *Recursive Symbolic Identity Architecture: A Complete Theoretical Framework*. Zenodo. <https://doi.org/xx.xxxx/zenodo.xxxxxxx>

This README will be kept in sync with the hash manifest to ensure provenance.
