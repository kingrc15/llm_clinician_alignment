from __future__ import annotations

VERIFY_SYSTEM = (
    "You are an expert heart failure cardiologist, ICU physician, and health systems reviewer. "
    "You judge whether a previously assigned readmission-reason bin is coherent with the full discharge "
    "documentation. Be succinct."
)


def make_verify_user_prompt(
    reason_for_readmission_binned: str,
    reason_for_readmission_binned_description: str,
    index_discharge_summary: str,
    readmit_discharge_summary: str,
) -> str:
    return (
        "A prior step assigned a single dominant readmission reason bin and a short justification, "
        "using only structured fields (not the full notes). You now have the actual discharge summaries.\n\n"
        "Task: Decide whether the chosen bin and its explanation are **consistent and defensible** given "
        "what is documented in the index (prior) and readmission discharge summaries. "
        "If the notes strongly support a different dominant mechanism, say so in your rationale.\n\n"
        "--- Assigned bin (verbatim) ---\n"
        f"{reason_for_readmission_binned.strip()}\n\n"
        "--- Bin explanation (verbatim) ---\n"
        f"{reason_for_readmission_binned_description.strip()}\n\n"
        "--- Index admission discharge summary ---\n"
        f"{index_discharge_summary.strip()}\n\n"
        "--- Readmission discharge summary ---\n"
        f"{readmit_discharge_summary.strip()}\n\n"
        "Output a JSON object with exactly these keys:\n"
        '- "makes_sense": boolean — true if the bin is a reasonable primary readmission driver given both notes; '
        "false if the notes clearly contradict it or another mechanism obviously dominates.\n"
        '- "rationale": string — brief (2–5 sentences) citing concrete details from the summaries.\n'
        '- "confidence": number between 0 and 1 — your confidence in this judgment (omit or null if unsure).\n'
        '- "qualifier": string — optional short note if judgment is borderline (e.g. \"partially\" right); '
        'use empty string \"\" if not applicable.'
    )
