"""Replace only propose(context) to add a future model-backed or local advisor."""
from copy import deepcopy
import importlib.util
from typing import Protocol
from .util import TuningError


class Agent(Protocol):
    def propose(self, context: dict) -> dict: ...


class DummyAgent:
    def propose(self, context):
        return {"space": deepcopy(context["original_space"]), "priority": [],
                "reason": "DummyAgent: retain configured domains; no reasoning service or automatic domain changes"}


def load_agent(config):
    if config["kind"] == "dummy":
        return DummyAgent()
    spec = importlib.util.spec_from_file_location("paretopilot_user_agent", config["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_agent(config.get("options", {}))


def validate_proposal(proposal, original):
    if not isinstance(proposal, dict) or not isinstance(proposal.get("space"), dict) or set(proposal["space"]) != set(original):
        raise TuningError("Agent must return the original parameter names")
    for name, values in proposal["space"].items():
        if not isinstance(values, list) or not values or any(type(v) is not int or v not in original[name] for v in values) or len(set(values)) != len(values):
            raise TuningError(f"Agent proposed unsupported values for {name}")
    if not isinstance(proposal.get("priority", []), list) or any(not isinstance(p, str) or p not in original for p in proposal.get("priority", [])):
        raise TuningError("Unknown Agent priority")
    return deepcopy(proposal)
