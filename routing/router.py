"""Who runs each step. Option 1 always answers "opus"; Option 2 asks Jev, then applies the gate.

Jev only ever picks among the three options below, which the harness prepares in advance.
"""

import json
from dataclasses import dataclass

from .clients import Jev
from .config import Settings, TIERS
from .steps import StepSpec

# The fixed option set. Descriptions are the only "knowledge" Jev has about each model: tune
# these first if routing looks off (and re-run, since they change the experiment).
OPTIONS = {
    "haiku": "Small, mechanical, well-specified work: boilerplate, short files, trivial edits. "
             "Low risk of subtle logic or cross-file consistency errors.",
    "sonnet": "Ordinary implementation work: moderate logic, a few files, or code that must match "
              "existing code closely. A reasonable default when unsure between the other two.",
    "opus": "Hard or high-risk work: intricate logic, many edge cases, ambiguous requirements, "
            "debugging an error that already survived a fix attempt, or very large context.",
}
INSTRUCTIONS = (
    "Pick the cheapest model that will complete this step correctly on the first attempt. "
    "A failed attempt costs a full retry on a stronger model, so do not pick a model that is too weak."
)


@dataclass
class Route:
    tier: str  # the tier the first attempt will use (after the confidence gate)
    jev_choice: str | None = None
    jev_confidence: float | None = None
    bumped: bool = False  # True if the low-confidence gate moved the pick up
    note: str = ""


class FixedRouter:
    """Option 1: every step goes to Opus. Makes no routing call."""

    def route(self, spec: StepSpec, signals: dict) -> Route:
        return Route("opus", note="option 1: fixed")


class JevRouter:
    def __init__(self, jev: Jev, cfg: Settings):
        self.jev, self.cfg = jev, cfg

    def route(self, spec: StepSpec, signals: dict) -> Route:
        # (a) describe the step: type, a short summary, and the signals Jev can weigh
        state = json.dumps({"step_key": spec.key, "step_type": spec.step_type,
                            "summary": spec.summary, "signals": signals})
        try:
            choice, conf = self.jev.choose(state, INSTRUCTIONS, OPTIONS, step=spec.key, step_type=spec.step_type)
        except Exception as e:  # network error, 4xx/5xx, schema drift: fail safe to the strongest model
            return Route("opus", note=f"jev error {type(e).__name__}: {e}")
        return self.gate(choice, conf)

    def gate(self, choice: str, conf: float) -> Route:
        """(c) Don't trust a low-confidence downgrade: move up one tier, or straight to Opus."""
        i = TIERS.index(choice)
        if conf < self.cfg.jev_min_confidence and i < len(TIERS) - 1:
            j = len(TIERS) - 1 if self.cfg.low_conf_policy == "opus" else i + 1
            return Route(TIERS[j], choice, conf, bumped=True, note=f"conf {conf:.2f} < {self.cfg.jev_min_confidence}")
        return Route(choice, choice, conf)
