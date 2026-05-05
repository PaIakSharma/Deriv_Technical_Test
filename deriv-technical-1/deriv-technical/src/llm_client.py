"""LLM client wrapping OpenRouter's OpenAI-compatible chat API.

Falls back to a deterministic mock when no OPENROUTER_API_KEY is configured,
so the pipeline always completes end-to-end. Both real and mock paths log
to ``llm_calls.jsonl`` with the same record shape.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from .utils import now_iso, sha256_hex

DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, log_path: str | Path = "llm_calls.jsonl") -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

        force_mock = os.environ.get("USE_MOCK_LLM", "").strip().lower() in ("1", "true", "yes")
        self.mock = force_mock or not self.api_key

        if self.mock and not force_mock:
            print(
                "[llm] OPENROUTER_API_KEY is not set — falling back to deterministic "
                "mock LLM. Set the key (or USE_MOCK_LLM=0) for real calls.",
                flush=True,
            )

        self.provider = "mock" if self.mock else "openrouter"
        self.log_path = Path(log_path)

    # ------------------------------------------------------------------ public
    def call(
        self,
        *,
        stage: str,
        strategy_id: str,
        prompt: str,
        system: Optional[str] = None,
        input_artifacts: Optional[List[str]] = None,
        output_artifact: Optional[str] = None,
        mock_response: Optional[Callable[[], str]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> str:
        """Make one LLM call and append a record to ``llm_calls.jsonl``."""
        prompt_hash = sha256_hex(prompt)

        if self.mock:
            if mock_response is None:
                raise LLMError(
                    f"stage={stage} strategy_id={strategy_id}: mock mode active "
                    "but no mock_response supplied."
                )
            response_text = mock_response()
            model_label = "mock-deterministic"
        else:
            response_text = self._call_openrouter(prompt, system, max_tokens, temperature)
            model_label = self.model

        record = {
            "stage": stage,
            "strategy_id": strategy_id,
            "timestamp": now_iso(),
            "provider": self.provider,
            "model": model_label,
            "prompt_hash": prompt_hash,
            "input_artifacts": list(input_artifacts or []),
            "output_artifact": output_artifact or "",
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return response_text

    # --------------------------------------------------------------- internals
    def _call_openrouter(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 — broad: provider can raise many types
                last_err = e
                wait = 2 ** attempt
                print(f"[llm] call failed (attempt {attempt + 1}/3): {e}; sleeping {wait}s", flush=True)
                time.sleep(wait)
        raise LLMError(f"OpenRouter call failed after 3 attempts: {last_err}")

    # ------------------------------------------------------------------- util
    def reset_log(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()
