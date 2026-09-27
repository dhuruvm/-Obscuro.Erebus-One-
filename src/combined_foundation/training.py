from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class TrainingRecipe:
    steps: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    reward_scale: float = 1.0
    verification_weight: float = 0.35
    grpo_weight: float = 0.25
    rlhf_weight: float = 0.2
    rlcd_weight: float = 0.2
    checkpoints: List[str] = field(default_factory=lambda: ["phase_1", "phase_2", "phase_3"])

    def summary(self) -> Dict[str, float]:
        return {
            "rlcd": self.rlcd_weight,
            "verification": self.verification_weight,
            "grpo_rlvr": self.grpo_weight,
            "rlhf": self.rlhf_weight,
            "learning_rate": self.learning_rate,
            "reward_scale": self.reward_scale,
        }


def simulate_rlcd_reward(prompt_quality: float, preference_diff: float) -> float:
    return (0.6 * prompt_quality + 0.4 * preference_diff)


def simulate_verification_reward(predicate_score: float, verifier_confidence: float) -> float:
    return 0.5 * predicate_score + 0.5 * verifier_confidence


def simulate_grpo_reward(task_score: float, consistency: float) -> float:
    return 0.7 * task_score + 0.3 * consistency


def simulate_rlhf_reward(helpfulness: float, safety: float) -> float:
    return 0.55 * helpfulness + 0.45 * safety


def run_demo_training(recipe: TrainingRecipe) -> Dict[str, float]:
    metrics = {}
    for step in range(recipe.steps):
        rlcd = simulate_rlcd_reward(0.78 + step * 0.01, 0.9)
        verification = simulate_verification_reward(0.84 + step * 0.008, 0.91)
        grpo = simulate_grpo_reward(0.8 + step * 0.012, 0.88)
        rlhf = simulate_rlhf_reward(0.82 + step * 0.01, 0.9)

        metrics[f"step_{step + 1}"] = {
            "RLCD": rlcd,
            "Verification": verification,
            "GRPO_RLVR": grpo,
            "RLHF": rlhf,
        }

    aggregated = {
        "rlcd": float(np.mean([v["RLCD"] for v in metrics.values()])),
        "verification": float(np.mean([v["Verification"] for v in metrics.values()])),
        "grpo_rlvr": float(np.mean([v["GRPO_RLVR"] for v in metrics.values()])),
        "rlhf": float(np.mean([v["RLHF"] for v in metrics.values()])),
        "combined_score": float(np.mean([
            np.mean([v["RLCD"], v["Verification"], v["GRPO_RLVR"], v["RLHF"]])
            for v in metrics.values()
        ])),
    }
    return aggregated
