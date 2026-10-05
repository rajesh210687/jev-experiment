"""Thin, metered clients for Opus (Anthropic SDK) and Jev (TypeSafe API, POST /v1/systemone).

Each network call is isolated in one small method (`_send` / `_post`) so it is easy to
stub out for a dry run.
"""

import os
import time

import anthropic
import httpx

from .config import FALLBACK_BETA, JEV, JEV_URL, OPUS, PRICES
from .ledger import CallRecord, Ledger, token_cost


class RefusalError(RuntimeError):
    pass


class Opus:
    def __init__(self, ledger: Ledger, option: str, run: int):
        self.client = anthropic.Anthropic()
        self.ledger, self.option, self.run = ledger, option, run

    def _send(self, **kwargs):
        return self.client.beta.messages.create(**kwargs)

    def create(self, step, *, messages, system, effort, model=OPUS, tools=None,
               json_schema=None, max_tokens=16000, note=""):
        output_config = {"effort": effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config=output_config,
            cache_control={"type": "ephemeral"},  # automatic prompt caching, same in both options
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if tools:
            kwargs["tools"] = tools

        t0 = time.perf_counter()
        resp = self._send(**kwargs)
        latency = time.perf_counter() - t0

        u = resp.usage
        # A refusal fallback can serve the turn on another model; price what actually ran.
        served_by = resp.model if resp.model in PRICES else model
        if resp.model != model:
            note = f"{note} served-by:{resp.model}".strip()
        rec = CallRecord(
            option=self.option, run=self.run, step=step, model=served_by,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            latency_s=latency,
            note=note,
        )
        rec.cost_usd = token_cost(served_by, rec.input_tokens, rec.output_tokens,
                                  rec.cache_write_tokens, rec.cache_read_tokens)
        self.ledger.add(rec)

        if resp.stop_reason == "refusal":
            raise RefusalError(f"{step}: request declined ({resp.stop_details})")
        return resp


class Jev:
    def __init__(self, ledger: Ledger, option: str, run: int):
        self.http = httpx.Client(
            timeout=30,
            headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"},
        )
        self.ledger, self.option, self.run = ledger, option, run

    def _post(self, payload: dict) -> dict:
        r = self.http.post(JEV_URL, json=payload)
        r.raise_for_status()
        return r.json()

    def decide(self, step: str, state: str, questions: dict) -> tuple[dict, CallRecord]:
        """Send typed questions; return (raw answers keyed by question name, ledger record)."""
        t0 = time.perf_counter()
        body = self._post({"model": JEV, "state": state, "questions": questions})
        latency = time.perf_counter() - t0

        usage = body.get("usage") or {}
        inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        cost = usage.get("cost")
        rec = CallRecord(
            option=self.option, run=self.run, step=step, model=JEV,
            input_tokens=inp, output_tokens=out, latency_s=latency,
            cost_usd=cost if cost is not None else token_cost(JEV, inp, out),
            questions=len(questions),
            note="" if cost is not None else "cost-estimated-from-price-table",
        )
        self.ledger.add(rec)

        if "answers" not in body:
            raise ValueError(
                "Unexpected Decisions API response shape (no 'answers' key; got "
                f"{sorted(body)}). Check the current response schema in the docs."
            )
        return body["answers"], rec
