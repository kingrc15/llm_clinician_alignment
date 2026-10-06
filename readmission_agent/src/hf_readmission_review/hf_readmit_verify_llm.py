from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from .hf_json_thinking import THINKING_BLOCK_INSTRUCTION, extract_first_thinking, extract_thinking_and_parse_json
from .hf_readmit_verify_prompts import VERIFY_SYSTEM, make_verify_user_prompt
from .json_utils import (
    build_retry_messages,
    complete_json_object,
    create_chat_completion,
    openai_assistant_text,
    parse_json_object,
)


class BinningSenseCheck(BaseModel):
    makes_sense: bool = Field(..., description="Whether the assigned bin fits the discharge summaries.")
    rationale: str = Field(..., description="Brief justification with references to the notes.")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence in the judgment.",
    )
    qualifier: str = Field(default="", description="Optional borderline / partial-fit note.")

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("qualifier", mode="before")
    @classmethod
    def _empty_qualifier(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


@dataclass(frozen=True)
class HfReadmitVerifyVllmBackend:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout_s: float = 600.0
    think_first: bool = True

    def _client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)

    def _system_prompt(self) -> str:
        if self.think_first:
            return VERIFY_SYSTEM + "\n\n" + THINKING_BLOCK_INSTRUCTION
        return (
            VERIFY_SYSTEM
            + "\n\nOutput exactly one JSON object with the required keys and no other text."
        )

    def _json_only_system(self) -> str:
        return (
            VERIFY_SYSTEM
            + "\n\nOutput exactly one JSON object with the required keys and no other text."
        )

    def verify_binning(
        self,
        reason_for_readmission_binned: str,
        reason_for_readmission_binned_description: str,
        index_discharge_summary: str,
        readmit_discharge_summary: str,
    ) -> Tuple[BinningSenseCheck, Dict[str, Any]]:
        client = self._client()
        user_content = make_verify_user_prompt(
            reason_for_readmission_binned=reason_for_readmission_binned,
            reason_for_readmission_binned_description=reason_for_readmission_binned_description,
            index_discharge_summary=index_discharge_summary,
            readmit_discharge_summary=readmit_discharge_summary,
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

        content = openai_assistant_text(resp.choices[0].message)
        raw: Dict[str, Any] = {
            "content": content,
            "usage": getattr(resp, "usage", None).model_dump() if getattr(resp, "usage", None) else None,
            "model": getattr(resp, "model", None),
            "think_first": self.think_first,
        }

        def _parse(text: str) -> BinningSenseCheck:
            if self.think_first:
                thinking, obj = extract_thinking_and_parse_json(text)
                raw["thinking"] = thinking
                return BinningSenseCheck.model_validate(obj)
            raw["thinking"] = None
            return BinningSenseCheck.model_validate(parse_json_object(text))

        try:
            pred = _parse(content)
        except (ValueError, ValidationError) as e:
            retry_tail = (
                "Keys: makes_sense (boolean), rationale (string), confidence (number 0-1, optional), "
                f'qualifier (string, use \"\" if none). Error: {e}'
            )
            if self.think_first:
                retry_tail = "First <thinking>...</thinking>, then one JSON object with " + retry_tail
            else:
                retry_tail = "Reply with exactly one JSON object with " + retry_tail
            retry_messages = build_retry_messages(
                messages,
                assistant_content=content,
                user_followup=f"Invalid JSON or schema. {retry_tail}",
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
                content3, resp3 = complete_json_object(
                    client,
                    model=self.model,
                    system_prompt=self._json_only_system(),
                    user_content=user_content,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw["content_json_fallback"] = content3
                raw["json_fallback"] = True
                raw["usage_json_fallback"] = (
                    getattr(resp3, "usage", None).model_dump() if getattr(resp3, "usage", None) else None
                )
                pred = BinningSenseCheck.model_validate(parse_json_object(content3))

        raw["verification"] = pred.model_dump()
        return pred, raw
