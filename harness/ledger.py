"""One record per model call; saved to calls.jsonl at the end of each run."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PRICES


@dataclass
class CallRecord:
    option: str  # "option1" | "option2"
    run: int
    step: str  # plan | context | route | execute | tests | recovery | fix
    model: str  # the model that actually served the call
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0  # includes thinking tokens on Claude
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    questions: int = 0  # decision calls: questions asked
    low_confidence: int = 0  # Jev calls: answers below threshold, re-asked to Opus
    note: str = ""


def token_cost(model: str, inp: int, out: int, cache_write: int = 0, cache_read: int = 0) -> float:
    if model not in PRICES:
        raise KeyError(f"No price for {model!r}; add it to PRICES in harness/config.py")
    p = PRICES[model]
    return (
        inp * p.input + out * p.output + cache_write * p.cache_write_5m + cache_read * p.cache_read
    ) / 1_000_000


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.records: list[CallRecord] = []
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, rec: CallRecord) -> CallRecord:
        self.records.append(rec)
        return rec

    def save(self) -> None:
        with self.path.open("w") as f:
            for rec in self.records:
                f.write(json.dumps(asdict(rec)) + "\n")
