from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class FactorLabel:
    label: str
    role: str
    confidence: int
    note: str = ""


@dataclass
class EvidenceSpan:
    selected_text: str
    note_type: str
    start_char: int
    end_char: int
    section_title: Optional[str] = None


@dataclass
class ClinicianReview:
    row_id: int
    clinician_email: str
    patient_identifier: str
    subject_id: str
    index_hadm_id: str
    readmit_hadm_id: str
    days_to_readmission: Optional[int]
    factors: List[FactorLabel] = field(default_factory=list)
    evidence_spans: List[EvidenceSpan] = field(default_factory=list)
    readmission_diagnoses: str = ""
    readmission_symptoms: str = ""
    overall_confidence: Optional[int] = None
    source_path: str = ""

    @property
    def selected_texts(self) -> List[str]:
        return [s.selected_text for s in self.evidence_spans if s.selected_text.strip()]

    @property
    def primary_factor(self) -> Optional[FactorLabel]:
        primaries = [f for f in self.factors if f.role == "primary"]
        return primaries[0] if primaries else None

    @property
    def factor_labels(self) -> List[str]:
        return [f.label for f in self.factors]


def _parse_review(path: Path) -> ClinicianReview:
    data = json.loads(path.read_text(encoding="utf-8"))
    case = data.get("case", {})
    payload = data.get("payload", {})
    meta = payload.get("caseMetadata", {})
    summary = payload.get("caseClinicalSummary", {})

    factors = [
        FactorLabel(
            label=str(f.get("label", "")),
            role=str(f.get("role", "")),
            confidence=int(f.get("confidence", 0)),
            note=str(f.get("note", "")),
        )
        for f in payload.get("factors", [])
    ]

    spans = [
        EvidenceSpan(
            selected_text=str(s.get("selectedText", "")),
            note_type=str(s.get("noteType", "")),
            start_char=int(s.get("startChar", 0)),
            end_char=int(s.get("endChar", 0)),
            section_title=s.get("sectionTitle"),
        )
        for s in payload.get("evidenceSpans", [])
    ]

    days = meta.get("daysToReadmission")
    return ClinicianReview(
        row_id=int(case.get("row_id", payload.get("caseId", 0))),
        clinician_email=str(data.get("clinician", {}).get("email", "")),
        patient_identifier=str(meta.get("patientIdentifier", "")),
        subject_id=str(case.get("subject_id", meta.get("subjectId", ""))),
        index_hadm_id=str(case.get("index_hadm_id", meta.get("indexHadmId", ""))),
        readmit_hadm_id=str(case.get("readmit_hadm_id", meta.get("readmitHadmId", ""))),
        days_to_readmission=int(days) if days is not None else None,
        factors=factors,
        evidence_spans=spans,
        readmission_diagnoses=str(summary.get("readmissionDiagnoses", "")),
        readmission_symptoms=str(summary.get("readmissionSymptoms", "")),
        overall_confidence=summary.get("overallConfidence"),
        source_path=str(path),
    )


def load_reviews(reviews_dir: str | Path) -> List[ClinicianReview]:
    root = Path(reviews_dir)
    reviews: List[ClinicianReview] = []
    for path in sorted(root.glob("**/*__submitted.json")):
        reviews.append(_parse_review(path))
    return reviews


def reviews_to_dataframe(reviews: List[ClinicianReview]):
    import pandas as pd

    rows = []
    for r in reviews:
        rows.append(
            {
                "row_id": r.row_id,
                "clinician_email": r.clinician_email,
                "patient_identifier": r.patient_identifier,
                "subject_id": r.subject_id,
                "index_hadm_id": r.index_hadm_id,
                "readmit_hadm_id": r.readmit_hadm_id,
                "days_to_readmission": r.days_to_readmission,
                "factor_labels": r.factor_labels,
                "primary_factor": r.primary_factor.label if r.primary_factor else None,
                "selected_texts": r.selected_texts,
                "n_spans": len(r.evidence_spans),
                "readmission_diagnoses": r.readmission_diagnoses,
                "readmission_symptoms": r.readmission_symptoms,
                "overall_confidence": r.overall_confidence,
                "source_path": r.source_path,
            }
        )
    return pd.DataFrame(rows)


def inter_rater_row_ids(reviews: List[ClinicianReview]) -> List[int]:
    from collections import Counter

    counts = Counter(r.row_id for r in reviews)
    return sorted(rid for rid, n in counts.items() if n > 1)


def iter_reviews(reviews_dir: str | Path) -> Iterator[ClinicianReview]:
    yield from load_reviews(reviews_dir)
