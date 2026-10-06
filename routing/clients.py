"""Metered clients: Claude (Anthropic SDK) and Jev (TypeSafe REST API).

Each network call lives in one tiny method (`_send` / `_post`) so a dry run can stub it.
"""

import os
import time

from .config import EFFORT, JEV_MODEL, JEV_URL, MODEL_IDS, TIERS
from .ledger import CallRecord, Ledger, token_cost

SYSTEM = (
    "You are a careful Python engineer working inside an automated pipeline. "
    "Follow the requested output format exactly. Standard library only for runtime code."
)


class Claude:
    def __init__(self, ledger: Ledger):
        import anthropic  # imported here so the dry run needs no SDK or key

        self.client = anthropic.Anthropic()
        self.ledger = ledger

    def _send(self, **kwargs):
        return self.client.messages.create(**kwargs)

    def call(self, tier: str, blocks: list[dict], *, step: str, step_type: str, attempt: int,
             max_tokens: int = 16000) -> tuple[str, str, CallRecord]:
        """Returns (text, stop_reason, record). The caller sets record.ok after validating."""
        kwargs = dict(model=MODEL_IDS[tier], max_tokens=max_tokens, system=SYSTEM,
                      messages=[{"role": "user", "content": blocks}])
        if EFFORT[tier]:
            kwargs["output_config"] = {"effort": EFFORT[tier]}
        # No `thinking` param: Opus 5.5 and Sonnet 5.5 run adaptive thinking by default and
        # Opus 5.5 rejects disabling it; Haiku 4.5 simply doesn't think unless asked.
        # No refusal fallbacks either: a silent fallback would change which model ran.
        t0 = time.perf_counter()
        resp = self._send(**kwargs)
        latency = time.perf_counter() - t0
        u = resp.usage
        rec = CallRecord(
            kind="model", step=step, step_type=step_type, attempt=attempt, tier=tier,
            model_id=MODEL_IDS[tier], input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
            cache_read_tokens=u.cache_read_input_tokens or 0, latency_s=latency,
        )
        rec.cost_usd = token_cost(tier, rec.input_tokens, rec.output_tokens,
                                  rec.cache_write_tokens, rec.cache_read_tokens)
        self.ledger.calls.append(rec)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.stop_reason, rec


class Jev:
    """One Choice question per routing call.

    UNVERIFIED: the request/response field names below follow the examples used by the earlier
    harness in this repo (POST /v1/systemone). I could not reach docs.typesafe.ai from the
    sandbox. Check them against TypeSafe's docs; everything Jev-specific is in `choose`.
    """

    def __init__(self, ledger: Ledger):
        import httpx

        self.http = httpx.Client(timeout=30, headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"})
        self.ledger = ledger

    def _post(self, payload: dict) -> dict:
        r = self.http.post(JEV_URL, json=payload)
        r.raise_for_status()
        return r.json()

    def choose(self, state: str, instructions: str, options: dict[str, str], *, step: str,
               step_type: str) -> tuple[str, float]:
        """Returns (label, confidence in 0..1) and logs the call, cost included."""
        payload = {
            "model": JEV_MODEL,
            "state": state,
            "questions": {"tier": {"type": "choice", "instructions": instructions, "criteria": options}},
        }
        t0 = time.perf_counter()
        body = self._post(payload)
        latency = time.perf_counter() - t0
        usage = body.get("usage") or {}
        inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        cost = usage.get("cost")  # use the API's own number when it reports one
        self.ledger.calls.append(CallRecord(
            kind="router", step=step, step_type=step_type, attempt=1, tier="jev", model_id=JEV_MODEL,
            input_tokens=inp, output_tokens=out, latency_s=latency, ok=None,
            cost_usd=cost if cost is not None else token_cost("jev", inp, out),
            note="" if cost is not None else "cost from price table",
        ))
        ans = body["answers"]["tier"]
        label, conf = ans["choice"], float(ans["confidence"])
        if label not in TIERS:
            raise ValueError(f"Jev returned {label!r}, not one of {TIERS}")
        return label, conf
