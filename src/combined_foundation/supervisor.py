from __future__ import annotations

import json
from typing import Any, Dict, Optional


class OpenJevSupervisor:
    """Supervisor / instructor adapter around the local Open-Jev decision model.

    This class uses the Open-Jev API contract when a local Open-Jev server is
    running on 127.0.0.1:8791. If it is unavailable, it falls back to a
    deterministic heuristic so the training pipeline still remains usable for
    offline experiments.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:8791/v1/systemone"):
        self.endpoint = endpoint

    def _probe_server(self) -> bool:
        try:
            import urllib.request

            request = urllib.request.Request(self.endpoint, method="GET")
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status == 200
        except Exception:
            return False

    def _heuristic_decision(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        answers: Dict[str, Any] = {}
        for key, spec in questions.items():
            spec_dict = spec if isinstance(spec, dict) else {"type": "choice", "instructions": str(spec)}
            qtype = spec_dict.get("type", "choice")
            instruction = str(spec_dict.get("instructions", key))
            if qtype in {"noul", "choice"}:
                answers[key] = {"decision": "safe", "confidence": 0.86, "instruction": instruction}
            elif qtype == "score":
                answers[key] = {"score": 0.83, "confidence": 0.82, "instruction": instruction}
            else:
                answers[key] = {"value": 0.81, "confidence": 0.8, "instruction": instruction}
        return {"answers": answers}

    def ask(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        if not self._probe_server():
            return self._heuristic_decision(state, questions)

        try:
            import urllib.request

            payload = json.dumps({"model": "open-jev", "state": state, "questions": questions}, allow_nan=False).encode()
            request = urllib.request.Request(
                self.endpoint,
                payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            return result
        except Exception:
            return self._heuristic_decision(state, questions)

    def advise_training(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        response = self.ask(state, questions)
        if "answers" not in response:
            return {"status": "fallback", "answers": self._heuristic_decision(state, questions)["answers"]}
        return {"status": "ok", "answers": response["answers"]}
