from __future__ import annotations

HF_ICU_READMIT_SYSTEM = (
    "You are an expert heart failure cardiologist, ICU physician, and health systems researcher. "
    "You reason about why patients with heart failure were readmitted to the ICU shortly after a prior "
    "hospital discharge. "
    "Use medical abbreviations and shorthand when clear (e.g. 'AIDS CD4 5 not on tx')."
)


def make_hf_readmit_user_prompt(
    patient_identifier: str,
    index_discharge_summary: str,
    readmit_discharge_summary: str,
    *,
    think_first: bool = True,
) -> str:
    body = (
        "The cases presented are patients who had an original admission for heart failure (HF) who were "
        "discharged then readmitted to the ICU for any cause within 30d of original discharge.\n\n"
        "For each patient case, use the previous discharge summary and the current discharge summary to "
        "reason about why that patient was readmitted. Consider yourself an expert heart failure "
        "cardiologist, ICU doctor, and health systems thinker.\n\n"
        "Provide a case-level clinical summary and one or more readmission factors. Each factor should "
        "capture a distinct reason the patient was readmitted. Mark exactly one factor as primary when "
        "possible (the single most dominant reason); other factors should be contributing.\n\n"
        "The output must be a JSON object with exactly these keys:\n"
        '- "patient_identifier": string (use the provided identifier)\n'
        '- "diagnosis": string (free text — readmission diagnoses you infer from the notes)\n'
        '- "symptoms": string (free text — presenting symptoms for the readmission)\n'
        '- "confidence": integer from 1 to 5 (overall confidence in your case assessment)\n'
        '- "factors": array of 1 or more objects, each with:\n'
        '    - "label": string (short free-text label for the factor, e.g. "Medication non-adherence")\n'
        '    - "role": string, either "primary" or "contributing"\n'
        '    - "confidence": integer from 1 to 5 (confidence in this specific factor)\n'
        '    - "clinical_notes": string (free text — clinical rationale or supporting notes)\n\n'
        "Patient identifier:\n"
        f"{patient_identifier.strip()}\n\n"
        "Previous (index HF admission) discharge summary:\n"
        f"{index_discharge_summary.strip()}\n\n"
        "Current (readmission) discharge summary:\n"
        f"{readmit_discharge_summary.strip()}\n"
    )
    if think_first:
        body += (
            "\n\nOutput format: first a single <thinking>...</thinking> block with your reasoning, "
            "then exactly one JSON object with the keys above (no markdown code fences around the JSON)."
        )
    return body
