from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI
from pydantic import ValidationError

from .hf_json_thinking import THINKING_BLOCK_INSTRUCTION, extract_first_thinking, extract_thinking_and_parse_json
from .hf_readmit_prompts import HF_ICU_READMIT_SYSTEM, make_hf_readmit_user_prompt
from .json_utils import (
    build_retry_messages,
    complete_json_object,
    create_chat_completion,
    openai_assistant_text,
    parse_json_object,
)


from .hf_readmit_schema import HeartFailureReadmitAnalysis


@dataclass(frozen=True)
class HfReadmitVllmBackend:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    timeout_s: float = 600.0
    # If True: model emits <thinking>...</thinking> then JSON (no response_format json_object).
    think_first: bool = True

    def _client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)

    def _system_prompt(self) -> str:
        if self.think_first:
            return HF_ICU_READMIT_SYSTEM + "\n\n" + THINKING_BLOCK_INSTRUCTION
        return (
            HF_ICU_READMIT_SYSTEM
            + "\n\nOutput exactly one JSON object with the required keys and no other text."
        )

    def _json_only_system(self) -> str:
        return (
            HF_ICU_READMIT_SYSTEM
            + "\n\nOutput exactly one JSON object with the required keys and no other text."
        )

    def analyze(
        self,
        patient_identifier: str,
        index_discharge_summary: str,
        readmit_discharge_summary: str,
    ) -> Tuple[HeartFailureReadmitAnalysis, Dict[str, Any]]:
        client = self._client()
        user_content = make_hf_readmit_user_prompt(
            patient_identifier=patient_identifier,
            index_discharge_summary=index_discharge_summary,
            readmit_discharge_summary=readmit_discharge_summary,
            think_first=self.think_first,
        )
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_content},
        ]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if not self.think_first:
            kwargs["response_format"] = {"type": "json_object"}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        try:
            resp = create_chat_completion(client, **kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            resp = create_chat_completion(client, **kwargs)

        msg0 = resp.choices[0].message
        content = openai_assistant_text(msg0)
        raw: Dict[str, Any] = {
            "content": content,
            "usage": getattr(resp, "usage", None).model_dump() if getattr(resp, "usage", None) else None,
            "model": getattr(resp, "model", None),
            "think_first": self.think_first,
        }

        def _parse(content_in: str) -> HeartFailureReadmitAnalysis:
            if self.think_first:
                thinking, obj = extract_thinking_and_parse_json(content_in)
                raw["thinking"] = thinking
                return HeartFailureReadmitAnalysis.model_validate(obj)
            raw["thinking"] = None
            obj = parse_json_object(content_in)
            return HeartFailureReadmitAnalysis.model_validate(obj)

        try:
            pred = _parse(content)
        except (ValueError, ValidationError) as e:
            retry_tail = (
                "patient_identifier (string), diagnosis (string), symptoms (string), "
                "confidence (integer 1-5), factors (array of 1+ objects each with "
                "label (string), role ('primary' or 'contributing'), confidence (integer 1-5), "
                "clinical_notes (string))."
            )
            if self.think_first:
                retry_tail = (
                    "First <thinking>...</thinking> with your reasoning, then exactly one JSON object with keys: "
                    + retry_tail
                )
            else:
                retry_tail = "Reply with exactly one JSON object using these keys: " + retry_tail + " No other text."
            retry_messages = build_retry_messages(
                messages,
                assistant_content=content,
                user_followup=f"Your last message was not valid. {retry_tail} Error: {e}",
            )
            kwargs2: Dict[str, Any] = {
                "model": self.model,
                "messages": retry_messages,
                "temperature": self.temperature,
            }
            if not self.think_first:
                kwargs2["response_format"] = {"type": "json_object"}
            if self.max_tokens is not None:
                kwargs2["max_tokens"] = self.max_tokens
            try:
                resp2 = create_chat_completion(client, **kwargs2)
            except Exception:
                kwargs2.pop("response_format", None)
                resp2 = create_chat_completion(client, **kwargs2)
            content2 = openai_assistant_text(resp2.choices[0].message)
            raw["content_retry"] = content2
            raw["usage_retry"] = (
                getattr(resp2, "usage", None).model_dump() if getattr(resp2, "usage", None) else None
            )
            try:
                pred = _parse(content2)
            except (ValueError, ValidationError) as e2:
                if not self.think_first:
                    raise e2 from e
                raw["thinking"] = extract_first_thinking(content) or extract_first_thinking(content2)
                json_user = make_hf_readmit_user_prompt(
                    patient_identifier=patient_identifier,
                    index_discharge_summary=index_discharge_summary,
                    readmit_discharge_summary=readmit_discharge_summary,
                    think_first=False,
                )
                content3, resp3 = complete_json_object(
                    client,
                    model=self.model,
                    system_prompt=self._json_only_system(),
                    user_content=json_user,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw["content_json_fallback"] = content3
                raw["json_fallback"] = True
                raw["usage_json_fallback"] = (
                    getattr(resp3, "usage", None).model_dump() if getattr(resp3, "usage", None) else None
                )
                obj = parse_json_object(content3)
                pred = HeartFailureReadmitAnalysis.model_validate(obj)

        raw["analysis"] = pred.model_dump()
        return pred, raw
