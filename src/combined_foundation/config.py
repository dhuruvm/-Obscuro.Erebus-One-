from dataclasses import dataclass, field
from typing import List


@dataclass
class CombinedFoundationConfig:
    model_name: str = "Obscuro-Erebus-Combined-Foundation-32B"
    target_params_b: float = 32.0  # 32 Billion Parameters
    hidden_size: int = 8192
    vocab_size: int = 64000
    num_layers: int = 48
    num_nar_steps: int = 8
    reasoning_depth: int = 32
    multimodal_experts: int = 16
    autoregressive_experts: int = 16
    top_k_experts: int = 4
    use_multimodal: bool = True
    use_extended_reasoning: bool = True
    use_nar: bool = True
    use_ar_moe: bool = True
    
    # Compute & Device Acceleration Strategy
    device_preference: str = "GPU"  # GPU -> CPU -> Auto
    use_mixed_precision: bool = True  # FP16/BF16 on GPU
    
    # Specialized Knowledge Domains
    knowledge_domains: List[str] = field(
        default_factory=lambda: [
            "fineweb",
            "human_philosophy",
            "psychology",
            "coding_systems",
            "mathematics_proofs",
            "quantum_physics",
            "chromium_self_browser",
        ]
    )

    # Autonomous Chromium Self-Learning Settings
    enable_chromium_learning: bool = True
    chromium_max_pages_per_session: int = 50
    chromium_min_param_threshold_b: float = 16.0

    # Training Objectives
    training_objectives: List[str] = field(
        default_factory=lambda: [
            "RLCD",
            "RL / Verification",
            "GRPO / RLVR",
            "RLHF",
        ]
    )

    @classmethod
    def create_32b_config(cls) -> "CombinedFoundationConfig":
        """Returns the targeted 32B Parameter Foundation Model configuration."""
        return cls(
            model_name="Obscuro-Erebus-Combined-Foundation-32B",
            target_params_b=32.0,
            hidden_size=8192,
            vocab_size=64000,
            num_layers=48,
            num_nar_steps=8,
            reasoning_depth=32,
            multimodal_experts=16,
            autoregressive_experts=16,
            top_k_experts=4,
        )

    def as_dict(self):
        return {
            "model_name": self.model_name,
            "target_params_b": self.target_params_b,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "num_nar_steps": self.num_nar_steps,
            "reasoning_depth": self.reasoning_depth,
            "multimodal_experts": self.multimodal_experts,
            "autoregressive_experts": self.autoregressive_experts,
            "top_k_experts": self.top_k_experts,
            "use_multimodal": self.use_multimodal,
            "use_extended_reasoning": self.use_extended_reasoning,
            "use_nar": self.use_nar,
            "use_ar_moe": self.use_ar_moe,
            "device_preference": self.device_preference,
            "use_mixed_precision": self.use_mixed_precision,
            "knowledge_domains": self.knowledge_domains,
            "enable_chromium_learning": self.enable_chromium_learning,
            "training_objectives": self.training_objectives,
        }

