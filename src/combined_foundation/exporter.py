from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from .config import CombinedFoundationConfig


def _base_dir() -> Path:
    """Return the directory next to the EXE (frozen) or the project root (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # In dev: this file is at src/combined_foundation/exporter.py
    # parents[2] is the project root (combined_foundation/)
    return Path(__file__).resolve().parents[2]


class ErebusExporter:
    """Export a C++ and GGUF-style model artifact for ErebusV1."""

    def __init__(self, config: CombinedFoundationConfig | None = None, out_dir: str | None = None):
        self.config = config or CombinedFoundationConfig(model_name="ErebusV1")
        base_dir = Path(out_dir) if out_dir is not None else _base_dir() / "exports"
        self.out_dir = base_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _cpp_code(self) -> str:
        return '''#include <cstdint>
#include <string>
#include <vector>

namespace obscuro {

struct ErebusV1Config {
    static constexpr int kHiddenSize = 4096;
    static constexpr int kVocabSize = 32000;
    static constexpr int kNumLayers = 24;
    static constexpr int kNumNarSteps = 8;
    static constexpr int kReasoningDepth = 12;
    static constexpr int kMultimodalExperts = 8;
    static constexpr int kAutoregressiveExperts = 8;
    static constexpr int kTopKExperts = 2;
};

class ErebusV1 {
public:
    explicit ErebusV1(const std::string& model_name = "ErebusV1")
        : model_name_(model_name) {}

    std::string model_name() const { return model_name_; }

    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> output(input.size(), 0.0f);
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = input[i];
        }
        return output;
    }

private:
    std::string model_name_;
};

}  // namespace obscuro
'''

    def _gguf_stub(self) -> str:
        return json.dumps({
            "gguf": "stub",
            "format": "GGUF",
            "model_name": self.config.model_name,
            "architecture": "CombinedFoundation",
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "num_layers": self.config.num_layers,
            "training_objectives": self.config.training_objectives,
            "notes": "This is an export manifest placeholder for ErebusV1. Replace with full GGUF tensor metadata in production training runs.",
        }, indent=2)

    def export_all(self) -> Dict[str, str]:
        cpp_path = self.out_dir / "ErebusV1.cpp"
        gguf_path = self.out_dir / "ErebusV1.gguf"
        gffu_path = self.out_dir / "ErebusV1.gffu"
        manifest_path = self.out_dir / "ErebusV1_manifest.json"

        cpp_path.write_text(self._cpp_code(), encoding="utf-8")
        gguf_path.write_text(self._gguf_stub() + "\n", encoding="utf-8")
        gffu_path.write_text(self._gguf_stub() + "\n", encoding="utf-8")
        manifest_path.write_text(json.dumps({
            "model_name": self.config.model_name,
            "exported_files": [
                str(cpp_path.name),
                str(gguf_path.name),
                str(gffu_path.name),
                str(manifest_path.name),
            ],
            "architecture": "CombinedFoundation",
        }, indent=2), encoding="utf-8")

        return {
            "cpp": str(cpp_path),
            "gguf": str(gguf_path),
            "gffu": str(gffu_path),
            "manifest": str(manifest_path),
        }
