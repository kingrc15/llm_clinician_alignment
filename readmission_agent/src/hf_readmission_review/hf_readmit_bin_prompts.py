from __future__ import annotations

import json
from typing import Any, Dict, List

# Exact strings the model must use for `reason_for_readmission_binned` (single dominating bucket).
READMISSION_BIN_LABELS: List[str] = [
    (
        "1. Premature discharge from hospital including discharge to the wrong location OR needed to have "
        "goals of care conversation but was not done adequately OR discharge to an acute care facility "
        "that was not able to care for patient per instructions from hospital"
    ),
    (
        "2. Medication non adherence OR fluid goal non adherence OR equipment non adherence (e.g., CPAP). "
        "Issues that may lead to this include language barriers, not able to refill medications, unable to "
        "pay for medications, inadequate social support, housing insecurity, etc."
    ),
    (
        "3. Inadequate prescription of meds at time of discharge (i.e., prescribed medications but they were "
        "not titrated appropriately OR had outpatient follow up a long time from discharge)"
    ),
    (
        "4. Inadequate escalation of care (i.e., patient decompensated at home and could not get proper "
        "escalation outpatient). Possibilities could be inadequate monitoring after discharge for factors "
        "like bowel movements, PT/INR levels, ensuring steroid was tapered and not suddenly stopped"
    ),
    "5. Miscellaneous (i.e., anything else that doesn't fit into the other bins)",
]

BINNING_SYSTEM = (
    "You are an expert heart failure cardiologist, ICU physician, and health systems thinker. "
    "You assign exactly one readmission reason bucket per case using only the structured information given. "
    "Be succinct."
)


def format_first_pass_context_from_analysis(analysis: Dict[str, Any]) -> str:
    """Build pass-2 context JSON from the new pass-1 analysis schema."""
    payload: Dict[str, Any] = {
        "diagnosis": analysis.get("diagnosis", ""),
        "symptoms": analysis.get("symptoms", ""),
        "confidence": analysis.get("confidence"),
        "factors": analysis.get("factors", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_first_pass_context(
    heart_failure_type: str,
    comorbidities: List[str],
    readmission_reasons_ranked: List[str],
    prevention_recommendations: str,
) -> str:
    payload: Dict[str, Any] = {
        "heart_failure_type": heart_failure_type,
        "comorbidities": comorbidities,
        "readmission_reasons_ranked": readmission_reasons_ranked,
        "prevention_recommendations": prevention_recommendations,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def make_binning_user_prompt(structured_case_json: str) -> str:
    bins_block = "\n".join(f"- {label}" for label in READMISSION_BIN_LABELS)
    return (
        "From the information, we need to bin the reason that each patient was readmitted into a single bucket. "
        "We must decide what the single most dominating reason for readmission was for each patient.\n\n"
        "Using the information provided place each admission into one of the following bins:\n"
        f"{bins_block}\n\n"
        "Consider yourself an expert heart failure cardiologist, ICU physician, and health systems thinker. "
        "Think deeply about the case and the context of the patient and the discharge and readmission notes. "
        "Please be succinct in text generated.\n\n"
        "Structured information from the prior analysis step (diagnosis, symptoms, confidence, factors):\n"
        f"{structured_case_json.strip()}\n\n"
        'Output a JSON object with exactly these keys: "reason_for_readmission_binned" and '
        '"reason_for_readmission_binned_description". '
        "The first field MUST be exactly one of the five bin strings listed above (copy it verbatim, including "
        "the leading number). "
        "The second field should be a tailored, succinct description of why that bin best fits this case."
    )
