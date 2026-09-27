from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, NoReturn, Optional

from .chromium_learner import ChromiumAutonomousLearner
from .config import CombinedFoundationConfig
from .industrial_runtime import IndustrialRuntime


class DockerAutomatedWorker:
    """Autonomous Docker worker daemon for continuous 32B foundation model training.

    Each run cycle:
    1. Runs an autonomous Chromium self-browser learning session (adds new RLVR corpus)
    2. Executes the full 4-phase industrial training pipeline
    3. Saves reproducible artifact checkpoints
    4. Sleeps for the configured interval before the next cycle

    Device priority: GPU (CUDA) → CPU (Fallback)
    """

    def __init__(
        self,
        interval_seconds: int = 300,
        chromium_pages_per_session: int = 20,
        model_name: str = "Obscuro-Erebus-Combined-Foundation-32B",
    ):
        self.interval_seconds = interval_seconds
        self.chromium_pages_per_session = chromium_pages_per_session
        self.model_name = model_name
        self.config = CombinedFoundationConfig.create_32b_config()
        self.runtime = IndustrialRuntime(model_name=model_name)
        self.learner = ChromiumAutonomousLearner(config=self.config)
        self.run_count = 0

    def _detect_device(self) -> str:
        """Detect the best available compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
                return f"CUDA GPU: {name} ({vram} GB VRAM)"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "Apple MPS (Metal)"
            else:
                return "CPU (Fallback)"
        except ImportError:
            return "CPU (PyTorch not loaded)"

    def _print_banner(self) -> None:
        device = self._detect_device()
        print("\n" + "=" * 70)
        print(f"  OBSCURO EREBUS 32B — AUTONOMOUS WORKER RUN #{self.run_count}")
        print(f"  Device  : {device}")
        print(f"  Model   : {self.model_name}")
        print(f"  Interval: {self.interval_seconds}s between runs")
        print("=" * 70)

    def run_chromium_phase(self) -> Dict[str, Any]:
        """Phase 0: Run autonomous Chromium self-learning to grow the corpus."""
        print(f"\n  [LEARNER] Autonomous Chromium Self-Browser Session (pages={self.chromium_pages_per_session})...")
        result = self.learner.run_autonomous_session(max_pages=self.chromium_pages_per_session)
        print(f"  [OK] Chromium Session Complete")
        print(f"       Pages Ingested  : {result['session_ingested_pages']}")
        print(f"       Total Corpus    : {result['total_chromium_corpus_records']} records")
        print(f"       Total RLVR Pairs: {result['total_chromium_rlvr_pairs']} pairs")
        return result

    def run_pipeline_phase(self) -> Any:
        """Phase 1-4: Run full industrial training pipeline."""
        print(f"\n  [PIPELINE] Executing Full 4-Phase Industrial Training Pipeline...")
        result = self.runtime.run()
        print(f"  [OK] Pipeline Complete")
        print(f"       Model     : {result.model_name}")
        print(f"       RLCD      : {result.metrics.get('RLCD', 0):.4f}")
        print(f"       RL_VER    : {result.metrics.get('RL_VERIFICATION', 0):.4f}")
        print(f"       GRPO_RLVR : {result.metrics.get('GRPO_RLVR', 0):.4f}")
        print(f"       RLHF      : {result.metrics.get('RLHF', 0):.4f}")
        print(f"       Final     : {result.final_score:.4f}")
        print(f"       Artifacts : {result.artifact_path}")
        return result

    def run_once(self) -> None:
        """Execute one full autonomous training cycle."""
        self.run_count += 1
        self._print_banner()

        try:
            # Phase 0: Chromium autonomous learning
            self.run_chromium_phase()

            # Phases 1-4: Full industrial pipeline
            result = self.run_pipeline_phase()

            print(f"\n  [DONE] Run #{self.run_count} completed. Score: {result.final_score:.4f}")

        except Exception as e:
            print(f"\n  [ERROR] Run #{self.run_count} failed: {e}")
            raise

        print("=" * 70 + "\n")

    def start_loop(self) -> NoReturn:
        """Start the infinite autonomous training loop."""
        print("\n" + "=" * 70)
        print("   OBSCURO EREBUS 32B — AUTONOMOUS SELF-TRAINING WORKER STARTED")
        print(f"   Training interval : {self.interval_seconds} seconds")
        print(f"   Chromium pages    : {self.chromium_pages_per_session} per session")
        print(f"   Model target      : {self.model_name}")
        print("   Press Ctrl+C to stop the worker cleanly.")
        print("=" * 70 + "\n")

        while True:
            try:
                self.run_once()
                print(f"  [SLEEP] Waiting {self.interval_seconds}s before next autonomous run...\n")
                time.sleep(self.interval_seconds)
            except KeyboardInterrupt:
                print("\n  [STOP] Worker interrupted by user. Exiting cleanly.")
                sys.exit(0)
            except Exception as e:
                print(f"  [RETRY] Error in worker run: {e}. Retrying in 30s...")
                time.sleep(30)


def run_worker(
    interval: int = 300,
    single_run: bool = False,
    chromium_pages: int = 20,
) -> None:
    """Entry point for the autonomous worker daemon."""
    worker = DockerAutomatedWorker(
        interval_seconds=interval,
        chromium_pages_per_session=chromium_pages,
    )
    if single_run:
        worker.run_once()
    else:
        worker.start_loop()
