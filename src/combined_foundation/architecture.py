import math
from typing import Dict, List

import numpy as np


class MoEExpert:
    def __init__(self, expert_id: int, hidden_size: int):
        self.expert_id = expert_id
        self.hidden_size = hidden_size
        self.effective_dim = min(hidden_size, 1024)
        self.weights = np.random.default_rng(expert_id).normal(0, 0.02, size=(self.effective_dim, self.effective_dim))

    def _project_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        input_dim = x.shape[-1]
        if input_dim == self.effective_dim:
            return x

        rng = np.random.default_rng(self.expert_id + 123)
        projector = rng.normal(0, 0.02, size=(self.effective_dim, input_dim))
        return x @ projector.T

    def forward(self, x: np.ndarray) -> np.ndarray:
        projected = self._project_input(x)
        out = projected @ self.weights.T
        if out.shape[-1] != self.effective_dim:
            out = out[:, : self.effective_dim]
        return out



class MultimodalMixtureOfExperts:
    def __init__(self, num_experts: int, hidden_size: int, top_k: int = 2):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.experts = [MoEExpert(i, hidden_size) for i in range(num_experts)]

    def route(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        scores = np.linspace(0.1, 1.0, self.num_experts, dtype=np.float32)
        normalized = scores / scores.sum()
        top_idx = np.argsort(normalized)[-self.top_k:]
        routed = {}
        score_map = {}
        for expert_id in top_idx:
            routed[str(expert_id)] = self.experts[int(expert_id)].forward(x)
            score_map[str(expert_id)] = float(normalized[int(expert_id)])
        return {"scores": np.asarray([score_map[str(i)] for i in top_idx], dtype=np.float32), "experts": routed}


class NonAutoregressiveBlock:
    def __init__(self, hidden_size: int, num_steps: int):
        self.hidden_size = hidden_size
        self.effective_dim = min(hidden_size, 1024)
        self.num_steps = num_steps

    def _project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] == self.effective_dim:
            return x
        rng = np.random.default_rng(42)
        projector = rng.normal(0, 0.02, size=(x.shape[-1], self.effective_dim))
        return x @ projector

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self._project(x)
        for _ in range(self.num_steps):
            h = h + 0.1 * np.tanh(h)
        return h


class ExtendedReasoningBlock:
    def __init__(self, hidden_size: int, reasoning_depth: int):
        self.hidden_size = hidden_size
        self.effective_dim = min(hidden_size, 1024)
        self.reasoning_depth = reasoning_depth

    def _project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] == self.effective_dim:
            return x
        rng = np.random.default_rng(7)
        projector = rng.normal(0, 0.02, size=(x.shape[-1], self.effective_dim))
        return x @ projector

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self._project(x)
        for i in range(self.reasoning_depth):
            h = h + (0.05 * (i + 1)) * np.sin(h)
        return h


class AutoregressiveMoEBlock:
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        self.hidden_size = hidden_size
        self.effective_dim = min(hidden_size, 1024)
        self.moe = MultimodalMixtureOfExperts(num_experts, hidden_size, top_k=top_k)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.effective_dim:
            rng = np.random.default_rng(99)
            projector = rng.normal(0, 0.02, size=(x.shape[-1], self.effective_dim))
            x = x @ projector
        routed = self.moe.route(x)
        out = np.zeros_like(x)
        for score, expert_out in zip(routed["scores"], routed["experts"].values()):
            out += score * expert_out
        return out


class CombinedFoundationModel:
    def __init__(self, config):
        self.config = config
        self.effective_dim = min(config.hidden_size, 1024)
        self.nar = NonAutoregressiveBlock(config.hidden_size, config.num_nar_steps)
        self.reasoning = ExtendedReasoningBlock(config.hidden_size, config.reasoning_depth)
        self.multimodal_moe = MultimodalMixtureOfExperts(config.multimodal_experts, config.hidden_size, top_k=config.top_k_experts)
        self.ar_moe = AutoregressiveMoEBlock(config.hidden_size, config.autoregressive_experts, top_k=config.top_k_experts)

    def _ensure_hidden(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[-1] == self.effective_dim:
            return x
        rng = np.random.default_rng(123)
        projector = rng.normal(0, 0.02, size=(x.shape[-1], self.effective_dim))
        return x @ projector


    def forward(self, text_tokens: np.ndarray, vision_tokens: np.ndarray | None = None) -> Dict[str, np.ndarray]:
        text = self._ensure_hidden(text_tokens)
        if vision_tokens is not None:
            vision = self._ensure_hidden(vision_tokens)
        else:
            vision = text.copy()

        nar_out = self.nar.forward(text)
        reasoning_out = self.reasoning.forward(nar_out)
        route_result = self.multimodal_moe.route(vision)
        mm_out = list(route_result["experts"].values())[0]
        ar_out = self.ar_moe.forward(reasoning_out)

        merged = 0.4 * reasoning_out + 0.3 * mm_out + 0.3 * ar_out
        return {
            "text": text,
            "nar_output": nar_out,
            "reasoning_output": reasoning_out,
            "multimodal_moe_output": mm_out,
            "autoregressive_moe_output": ar_out,
            "combined_output": merged,
        }

    def estimate_parameters(self) -> int:
        vocab_embed = self.config.vocab_size * self.config.hidden_size * 2
        layer_params = self.config.num_layers * (
            4 * self.config.hidden_size * self.config.hidden_size
            + 8 * self.config.hidden_size * self.config.hidden_size
        )
        nar_params = self.config.hidden_size * self.config.hidden_size * self.config.num_nar_steps
        reasoning_params = self.config.hidden_size * self.config.reasoning_depth
        mm_moe_params = self.config.multimodal_experts * self.config.hidden_size * self.config.hidden_size
        ar_moe_params = self.config.autoregressive_experts * self.config.hidden_size * self.config.hidden_size

        total = vocab_embed + layer_params + nar_params + reasoning_params + mm_moe_params + ar_moe_params
        return total

