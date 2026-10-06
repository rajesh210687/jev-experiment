"""One CallRecord per API call (Claude and Jev), one StepRecord per step."""

from dataclasses import asdict, dataclass, field

from .config import PRICES


@dataclass
class CallRecord:
    kind: str  # "model" (a Claude step call) | "router" (a Jev routing call)
    step: str  # step key, e.g. "core"
    step_type: str  # plan | codegen | select_tests | fix_errors
    attempt: int  # 1 = first try; >1 = a retry or escalation. Router calls are always 1.
    tier: str  # haiku | sonnet | opus | jev
    model_id: str
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0  # includes thinking tokens
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    ok: bool | None = None  # model calls: did the output pass the step's checks?
    note: str = ""


@dataclass
class StepRecord:
    step: str
    step_type: str
    signals: dict = field(default_factory=dict)  # what the router was told about the step
    jev_choice: str | None = None  # Jev's raw pick (None in Option 1 or on a Jev error)
    jev_confidence: float | None = None
    first_tier: str = ""  # tier of the first attempt, after the confidence gate
    final_tier: str = ""  # tier that produced the accepted output ("" if the step failed)
    low_confidence_bump: bool = False
    escalations: list[str] = field(default_factory=list)  # "haiku->sonnet: <reason>"
    escalation_cost_usd: float = 0.0  # spend on attempts that were then escalated away from
    same_tier_retries: int = 0  # retries at Opus, or in Option 1 (no higher tier exists)
    attempts: int = 0
    ok: bool = False


def token_cost(tier: str, inp: int, out: int, cache_write: int = 0, cache_read: int = 0) -> float:
    p = PRICES[tier]
    return (inp * p.input + out * p.output + cache_write * p.cache_write_5m + cache_read * p.cache_read) / 1e6


class Ledger:
    def __init__(self) -> None:
        self.calls: list[CallRecord] = []
        self.steps: list[StepRecord] = []

    def to_dict(self) -> dict:
        return {"calls": [asdict(c) for c in self.calls], "steps": [asdict(s) for s in self.steps]}
