from __future__ import annotations

from typing import Any, Dict


class Phi3MiniBackend:
    """Adapter for the Microsoft Phi-3 Mini base model family.

    This keeps the project usable even without a live model download by exposing
    a structured description and a safe loader that degrades gracefully if the
    model cannot be fetched or transformers is unavailable.
    """

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", max_context_length: int = 4096):
        self.model_name = model_name
        self.max_context_length = max_context_length

    def describe(self) -> Dict[str, Any]:
        try:
            import transformers  # noqa: F401
            status = "transformers-ready"
        except Exception:
            status = "fallback-mode"

        return {
            "base_model": self.model_name,
            "family": "Phi-3 Mini",
            "context_length": self.max_context_length,
            "status": status,
            "note": "Compatible with Microsoft Phi-3 Mini instruction-tuned checkpointing and wrapper training.",
        }

    def load(self) -> Dict[str, Any]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - runtime fallback
            return {"status": "fallback", "error": f"transformers unavailable: {exc}"}

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            return {
                "status": "loaded",
                "tokenizer": tokenizer,
                "model": model,
                "base_model": self.model_name,
            }
        except Exception as exc:  # pragma: no cover - runtime fallback
            return {"status": "fallback", "error": f"model load failed: {exc}"}
