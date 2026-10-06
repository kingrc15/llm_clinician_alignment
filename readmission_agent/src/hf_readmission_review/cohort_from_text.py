"""
Build a one-row cohort parquet from plain-text discharge summaries.

Used by ``run_pair_review`` so the same three-pass pipeline as the full cohort
tooling can run without MIMIC-derived cohort files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_minimal_cohort_parquet(
    *,
    out_path: str | Path,
    row_id: int,
    patient_identifier: str,
    index_discharge_summary: str,
    readmit_discharge_summary: str,
    readmit_admission_note: str = "",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "row_id": int(row_id),
                "patient_identifier": str(patient_identifier),
                "index_discharge_summary": str(index_discharge_summary),
                "readmit_discharge_summary": str(readmit_discharge_summary),
                "readmit_admission_note": str(readmit_admission_note or ""),
            }
        ]
    )
    df.to_parquet(out_path, index=False)
    return out_path
