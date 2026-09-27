"""Optional adapter layer for NVIDIA NeMo compatibility.

This intentionally stays lightweight so the project can run without a full
NeMo installation while still exposing the integration points expected in an
NVIDIA stack-based workflow.
"""

from __future__ import annotations

from typing import Any, Dict


class NeMoAdapter:
    """Thin compatibility wrapper around core model lifecycle hooks."""

    def __init__(self, model_name: str = "Obscuro-Erebus-Combined-Foundation"):
        self.model_name = model_name

    def build_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "framework": "NeMo",
            "stack": [
                "multimodal_moe",
                "autoregressive_moe",
                "reasoning_layers",
                "RLCD",
                "verification_rl",
                "grpo_rlvr",
                "rlhf",
            ],
        }

    def export_run_spec(self) -> Dict[str, Any]:
        return {
            "trainer": "NeMo Trainer",
            "precision": "bf16",
            "distributed": "ddp",
            "model_type": "custom_combined_foundation",
        }
