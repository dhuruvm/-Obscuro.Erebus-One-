from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from .pipeline import IndustrialMultiModalTrainingPipeline


def _base_dir() -> Path:
    """Return the directory next to the EXE (frozen) or project root (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


@dataclass
class IndustrialRunResult:
    model_name: str
    dataset_info: Dict[str, Any]
    system_info: Dict[str, Any]
    stages: List[str]
    metrics: Dict[str, float]
    final_score: float
    artifact_path: str


class IndustrialRuntime:
    """Production runtime executing phased pipeline and saving reproducible artifacts."""

    def __init__(self, model_name: str = "Obscuro-Erebus-Combined-Foundation-32B", artifact_root: str | None = None):
        self.model_name = model_name
        self.pipeline = IndustrialMultiModalTrainingPipeline(model_name=model_name)
        self.artifact_root = Path(artifact_root or _base_dir() / "artifacts" / "runs")
        self.artifact_root.mkdir(parents=True, exist_ok=True)


    def run(self) -> IndustrialRunResult:
        summary = self.pipeline.run_full_pipeline()
        run_dir = self.artifact_root / "industrial_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_name": summary.model_name,
            "dataset_info": summary.dataset_info,
            "system_info": summary.system_info,
            "stages": summary.stages,
            "metrics": summary.metrics,
            "final_score": float(summary.final_score),
        }

        artifact_path = run_dir / "pipeline_summary.json"
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return IndustrialRunResult(
            model_name=summary.model_name,
            dataset_info=summary.dataset_info,
            system_info=summary.system_info,
            stages=summary.stages,
            metrics=summary.metrics,
            final_score=float(summary.final_score),
            artifact_path=str(artifact_path),
        )
