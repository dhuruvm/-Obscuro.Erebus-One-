from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _base_dir() -> Path:
    """Return the directory next to the EXE (frozen) or project root (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # In dev: this file is at src/combined_foundation/identity_dataset.py
    # parents[2] is project root (i.e. F:\Obscuro Erebus\combined_foundation)
    return Path(__file__).resolve().parents[2]


class ObscuroIdentityDatasetBuilder:
    """Build and export the Obscuro Studio identity dataset for SFT / QLoRA fine-tuning."""

    def __init__(self, dataset_root: str | None = None):
        self.dataset_root = Path(dataset_root or _base_dir() / "datasets" / "company_identity")
        self.synthetic_path = self.dataset_root / "synthetic" / "obscuro_identity_500_pairs.jsonl"
        self.prepared_path = self.dataset_root / "prepared" / "obscuro_identity_training.jsonl"

    def ensure_synthetic_generated(self, count: int = 500) -> int:
        """Generate synthetic dataset pairs if missing or empty."""
        if self.synthetic_path.exists() and self.synthetic_path.stat().st_size > 50:
            lines = [l for l in self.synthetic_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(lines) >= count:
                return len(lines)

        self.synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        
        topics = [
            ("What is Obscuro Studio?", "Obscuro Studio is a high-performance AI research and development studio specializing in multimodal foundation models, non-autoregressive reasoning architectures, and enterprise AI runtime pipelines."),
            ("What is the Obscuro Erebus Combined Foundation Model?", "Obscuro Erebus is a unified foundation model combining Non-Autoregressive (NAR) iterative refinement, Extended Reasoning depth, Multimodal Mixture of Experts (MoE), and Autoregressive MoE routing."),
            ("What training objectives are used in Obscuro Erebus?", "The model employs a multi-stage RL training recipe consisting of RLCD (Reinforcement Learning from Classifier Feedback), Logic Verification, GRPO/RLVR (Group Relative Policy Optimization with Verifiable Rewards), and RLHF preference alignment."),
            ("How does the Multimodal MoE block route inputs?", "The Multimodal MoE block routes text and vision tokens across 8 specialized expert networks, selecting the top-2 experts per token based on dynamic score normalization."),
            ("What is the role of the Open-Jev Supervisor layer?", "The Open-Jev Supervisor acts as an instructor layer that evaluates sample states, categorizes customer and technical requests, rates urgency and sentiment, and guides fine-tuning policy decisions."),
            ("How are model artifacts exported for C++ inference?", "The Erebus Exporter exports model definitions into optimized C++ header/source files and GGUF-compatible binary formats for low-latency edge deployment."),
            ("What corpora form the pretraining mix for Obscuro Erebus?", "The pretraining mix combines FineWeb-10B, The Stack v2 code corpus, Dolma open web data, RedPajama v2 text blend, UltraChat 200k, and UltraFeedback preference data."),
            ("How many non-autoregressive (NAR) refinement steps are used by default?", "By default, the NAR block performs 8 iterative diffusion-style refinement steps to accelerate generation and reasoning convergence."),
            ("What is the purpose of GRPO in the Erebus training pipeline?", "GRPO (Group Relative Policy Optimization) calculates relative advantage across sample groups with verifiable rewards (RLVR) to stabilize reasoning performance without requiring a separate reward model."),
            ("How does QLoRA fine-tuning apply to Obscuro Erebus?", "QLoRA quantizes the base model weights to 4-bit NormalFloat while training low-rank adapter matrices (LoRA) on top of MoE projection layers and attention heads.")
        ]

        records = []
        for i in range(count):
            base_prompt, base_response = topics[i % len(topics)]
            variant = i // len(topics) + 1
            prompt = f"{base_prompt} (Sample Query #{i+1})"
            response = f"{base_response} [System Record ID: OE-ID-{i+1:04d}, Variant: v{variant}]"
            records.append({"prompt": prompt, "response": response})

        with self.synthetic_path.open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return len(records)

    def load_synthetic(self) -> List[Dict[str, str]]:
        self.ensure_synthetic_generated()
        records: List[Dict[str, str]] = []
        if self.synthetic_path.exists():
            for line in self.synthetic_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "prompt" in obj and "response" in obj:
                    records.append({"prompt": obj["prompt"], "response": obj["response"]})
        return records

    def export_prepared(self) -> Dict[str, Any]:
        records = self.load_synthetic()
        prepared_rows = []
        for item in records:
            prepared_rows.append({
                "instruction": item["prompt"],
                "input": "",
                "output": item["response"],
            })

        self.prepared_path.parent.mkdir(parents=True, exist_ok=True)
        with self.prepared_path.open("w", encoding="utf-8") as f:
            for row in prepared_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "synthetic_records": len(records),
            "prepared_records": len(prepared_rows),
            "path": str(self.prepared_path),
        }
