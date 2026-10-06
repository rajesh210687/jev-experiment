"""Models, prices (each with the date it was checked), and every tunable knob."""

from dataclasses import dataclass

TIERS = ["haiku", "sonnet", "opus"]  # cheapest -> strongest; list index = tier rank
MODEL_IDS = {
    "haiku": "claude-haiku-4-5",  # alias of the dated snapshot claude-haiku-4-5-20251001
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}
# Same effort on every tier that supports it, so effort doesn't confound the comparison.
# Haiku 4.5 does not support the `effort` parameter, so it gets none.
EFFORT = {"haiku": None, "sonnet": "medium", "opus": "medium"}

# Jev, called directly on TypeSafe's API. UNVERIFIED wire format: see the guide, Step 1.
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"  # pin a specific version from TypeSafe's docs before a real experiment


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cache_write_5m: float
    cache_read: float
    source: str
    checked: str  # the date the price was looked up


_ANTHROPIC = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICES = {
    "haiku": Price(1.00, 5.00, 1.25, 0.10, _ANTHROPIC, "2026-10-06"),
    "sonnet": Price(2.00, 10.00, 2.50, 0.20, _ANTHROPIC, "2026-10-06"),
    "opus": Price(4.00, 20.00, 5.00, 0.20, _ANTHROPIC, "2026-10-06"),
    # Jev: $0.042 per MTok input, output free. Reported by third-party pages and OpenRouter's
    # "Jev 1.13" listing; TypeSafe's own pricing page was not reachable. Confirm in your console.
    "jev": Price(0.042, 0.0, 0.0, 0.0, "third-party listings; confirm in TypeSafe console", "2026-10-06"),
}


@dataclass
class Settings:
    granularity: str = "fine"  # "fine" (7 code-gen sub-steps) | "coarse" (4 groups)
    # Option 2 only. Below this Jev confidence, a downgrade is not trusted.
    jev_min_confidence: float = 0.75
    low_conf_policy: str = "one_tier"  # "one_tier": go one tier up | "opus": go straight to Opus
    max_attempts_per_step: int = 3  # first try + escalations/retries; identical for both options
    max_fix_rounds: int = 3  # select-tests -> run -> fix rounds before giving up
    cache: bool = True  # put cache_control on the shared prompt prefix (same in both options)
