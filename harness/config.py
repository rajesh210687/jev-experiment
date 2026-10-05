"""Model IDs, prices, and decision thresholds. Everything you would tune lives here."""

from dataclasses import dataclass

OPUS = "claude-opus-5-5"

# Jev is served through OpenRouter's Decisions API (alpha). Pin the version:
# confidence thresholds below are tuned against one specific Jev release.
JEV = "typesafe/jev-1.13"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MAX_STATE_CHARS = 100_000  # Jev's request budget is ~32K tokens (~150K chars); stay well under
JEV_MAX_QUESTIONS_PER_CALL = 16  # UNVERIFIED: check the Decisions API docs for the real per-request limit

# Server-side refusal fallback for Opus 5.5. If a turn is ever served by another model,
# the ledger prices it by the model that actually ran (response.model).
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cache_write_5m: float
    cache_read: float
    source: str
    checked: str  # date the price was looked up


_ANTHROPIC = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICES = {
    "claude-opus-5-5": Price(4.00, 20.00, 5.00, 0.20, _ANTHROPIC, "2026-10-05"),
    # Only used if you pass --cheap-model claude-sonnet-5-5 to Option 2.
    "claude-sonnet-5-5": Price(2.00, 10.00, 2.50, 0.20, _ANTHROPIC, "2026-10-05"),
    # Refusal-fallback targets; they only appear if Opus 5.5 declines a request.
    "claude-opus-5": Price(5.00, 25.00, 6.25, 0.50, _ANTHROPIC, "2026-10-05"),
    "claude-opus-4-8": Price(5.00, 25.00, 6.25, 0.50, _ANTHROPIC, "2026-10-05"),
    # Jev: $0.042/MTok input, output free. The ledger prefers the `usage.cost` the
    # Decisions API returns and only falls back to this rate if that field is absent.
    JEV: Price(0.042, 0.0, 0.0, 0.0, "https://openrouter.ai/docs/guides/community/jev", "2026-10-05"),
}

# Effort per step. Opus 5.5 defaults to "medium", so always set it explicitly.
EFFORT = {
    "plan": "high",
    "execute_complex": "high",
    "execute_simple": "low",
    "decide": "low",  # Option 1's small decisions, and Option 2's Opus fallbacks
}

# Option 2: accept Jev's answer only at or above this confidence, otherwise re-ask Opus.
# Noul has no confidence field; the harness uses |2p - 1| (0 at p=0.5, 1 at p=0 or 1).
JEV_MIN_CONFIDENCE = {
    "context": 0.6,  # which files to include: a wrong "no" costs a re-read, cheap to recover
    "route": 0.7,  # simple vs complex: a wrong "simple" can cost a failed step
    "tests": 0.6,  # which tests run first: the full suite still runs afterwards
    "recovery": 0.8,  # which recovery path: a wrong path burns an Opus fix call
}

MAX_TOOL_TURNS = 8  # executor tool-use turns per subtask
MAX_FIX_ATTEMPTS = 3  # recovery attempts per subtask
PYTEST_TIMEOUT_S = 120
