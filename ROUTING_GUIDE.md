# One model vs. Jev-routed models: a cost-comparison harness

This guide builds a small Python harness (about 1,000 lines, no frameworks) that runs one coding task four ways and reports what each cost:

| | Fine split (7 code-generation sub-steps) | Coarse split (4 groups) |
|---|---|---|
| **Option 1**: Claude Opus 5.5 does every step | `opt1-fine` | `opt1-coarse` |
| **Option 2**: Jev picks Haiku 4.5 / Sonnet 5.5 / Opus 5.5 per step | `opt2-fine` | `opt2-coarse` |

The code in this guide is the code in `routing/` in this repo, pasted in verbatim. Run it with `python run_routing.py`.

## Read this first: what is verified and what is not

**Tested.** I ran the whole harness end to end with both APIs stubbed (`python -m routing.dry_run`): all four configurations, 3 repeats each, 12/12 runs pass the held-out acceptance tests. The stub injects failures on purpose (malformed output, a renamed function, a failing test, a fix that fixes nothing), so the escalation, confidence-gate, and fix paths all execute. **It has not been run against the live Claude or Jev APIs**, because this session has no API keys. Every dollar figure in the example table at the end comes from the stub, so it is fake. It shows the report's format, not results.

**Verified from Anthropic's docs on 2026-10-06** ([pricing](https://platform.claude.com/docs/en/about-claude/pricing), [models overview](https://platform.claude.com/docs/en/models/overview)):

| Model | API ID | Input $/MTok | Output $/MTok | 5m cache write | Cache read |
|---|---|---:|---:|---:|---:|
| Claude Haiku 4.5 | `claude-haiku-4-5` (alias of the dated snapshot `claude-haiku-4-5-20251001`) | 1.00 | 5.00 | 1.25 | 0.10 |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | 2.00 | 10.00 | 2.50 | 0.20 |
| Claude Opus 5.5 | `claude-opus-5-5` | 4.00 | 20.00 | 5.00 | 0.20 |
| Jev | `jev-latest` (placeholder, see below) | 0.042 | free | n/a | n/a |

**Not verified: Jev.**
- I could not open `docs.typesafe.ai` (the sandbox's network proxy blocks it), so **I don't know Jev's exact SDK method names, request fields, or response fields**, and I did not use a Jev SDK. The harness calls TypeSafe's REST endpoint with `httpx`, in the field layout the earlier harness in this repo used (`POST https://api.typesafe.ai/v1/systemone`, one `choice` question, answer with `choice` and `confidence`). Treat that layout as **UNVERIFIED** and check it against [docs.typesafe.ai](https://docs.typesafe.ai). Everything Jev-specific is isolated in `Jev.choose` and `Jev._post` in `routing/clients.py`.
- The Jev price of **$0.042 per million input tokens, output free** comes from third-party pages that quote TypeSafe ([eesel](https://www.eesel.ai/blog/typesafe-jev-pricing), [Layer3 Labs](https://www.layer3labs.io/guides/jev-pricing), and OpenRouter's [Jev 1.13 listing](https://openrouter.ai/typesafe/jev-1.13)), found 2026-10-06. I could not read TypeSafe's own pricing page. Confirm it in your TypeSafe console. The harness uses the `usage.cost` value from Jev's response if there is one, and otherwise prices the call from this table (the ledger notes which).
- The Jev model ID `jev-latest` is a placeholder. Pin the exact version from TypeSafe's docs before a real experiment so a model update can't change your results mid-run.

**Sample task.** Your prompt left the task as a placeholder, so I used the example you gave: a CLI that parses a CSV of orders and flags duplicates, with unit tests. It is specified in `task/TASK.md`, with 11 held-out acceptance tests in `task/acceptance/` (from the earlier harness in this repo). Swap in your own task by editing those files and the `LAYOUT` / step briefs in `routing/steps.py`.

---

## 1. Setup: SDKs, keys, project structure

You need Python 3.11+, the Anthropic SDK, `httpx` (for Jev's REST API), and `pytest` (the agent's code is tested with it, and so is the quality gate). `requirements.txt` already has all three.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                  # anthropic>=1.11,<2  httpx>=0.27  pytest>=8
cp .env.example .env                             # then fill in both keys
set -a && source .env && set +a
```

```bash
# .env.example
ANTHROPIC_API_KEY=sk-ant-...      # Claude Console -> API keys
TYPESAFE_API_KEY=...              # console.typesafe.ai/keys
```

```text
jev-experiment/
├── run_routing.py            # entry point
├── routing/
│   ├── config.py             # models, prices + dates, tunable settings
│   ├── ledger.py             # one record per API call, one per step
│   ├── clients.py            # metered Claude client and Jev client
│   ├── steps.py              # the fixed step sequence (fine / coarse)
│   ├── checks.py             # parse + validate model output (what triggers escalation)
│   ├── workspace.py          # project sandbox: snapshot/restore, pytest
│   ├── router.py             # Option 1: always Opus. Option 2: Jev + confidence gate
│   ├── agent.py              # the agent loop both options share
│   ├── runner.py             # the 4-configuration x N-repeat matrix
│   ├── report.py             # the comparison tables
│   └── dry_run.py            # same harness with both APIs stubbed
├── harness/                  # earlier harness; routing/ reuses its Workspace + acceptance gate
└── task/                     # TASK.md, starter/, acceptance/ (held-out tests)
```

> **Safety:** the agent writes Python that the harness then executes (imports, `python -m`, pytest). Run it in a container or throwaway VM, not on a machine with credentials you care about.

## 2. Configuration: models, prices, knobs

All prices carry their source and the date they were checked, and the report prints them. Opus and Sonnet get the same `effort` so effort doesn't confound the comparison; Haiku 4.5 doesn't support the `effort` parameter at all, which is itself a quality difference to keep in mind. Your two tunables for Option 2 are `jev_min_confidence` and `low_conf_policy` (Step 7).

```python
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
```

## 3. The ledger: tokens, calls, latency, cost

Every Claude call and every Jev call appends a `CallRecord` (step, attempt number, model, input/output/cache tokens, latency, cost, and whether the output passed checks). Every step appends a `StepRecord` (what Jev was told, what it chose, its confidence, which model first ran, which model's output was accepted, escalations). The report is computed from these two lists, so nothing is estimated after the fact.

```python
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
```

## 4. Clients: Claude and Jev

Both clients isolate the network call in one tiny method (`_send`, `_post`) so the dry run can stub it.

- **Claude:** one plain `messages.create` per step, no tools. Cost is computed from `usage`, including cache read/write tokens, at the price of the tier that actually ran. I deliberately did **not** enable the server-side refusal `fallbacks` parameter: a silent fallback to another model would change which model ran and break the accounting. A `refusal` stop reason is instead treated as a failed output, which escalates like any other failure.
- **Jev:** one `Choice` question per routing call. The response's `usage` is logged as its own `router` call so Jev's cost and latency are counted like any other. If Jev errors, the router fails safe to Opus.

```python
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
```

## 5. The step sequence (shared by both options) and the granularity flag

Both options walk the same fixed sequence:

```
plan -> code generation -> [ select tests -> run -> fix errors ]  (repeat until green or out of rounds)
```

- **`--granularity {fine,coarse,both}`** (default `both`, which fills the four-configuration matrix) sets `Settings.granularity`, the code-generation split. **Fine** = 7 sub-steps: scaffold, models, core, edge_cases, cli (I/O + wiring), unit_tests, docs_cleanup. **Coarse** = 4 groups built by merging those same specs: `scaffold_models`, `core_edge_cases`, `cli_unit_tests`, `docs_cleanup`. Coarse is generated from the fine specs (`_merge`), so the work is identical and only the call boundaries move.
- Planning, test selection, and error fixing are their own steps in both settings.
- **The file layout is fixed by the harness**, not chosen by the model, so every configuration builds the same file set. The plan step designs the contents: function signatures per file, edge cases, test cases, and the CLI contract.
- **Each step gets only what it needs** (see `code_context` and `plan_slice` in Step 8): the slice of the plan for the files it writes or depends on, the *full source* of the files it must stay consistent with (`needs_full`), and only the *signatures* (an AST digest) of every other file. This keeps later sub-steps consistent with earlier ones without re-sending the whole codebase each time.

```python
"""The fixed step sequence, shared by both options.

plan -> code generation (7 sub-steps "fine", or 4 groups "coarse") -> [select tests -> run -> fix]*

The file layout is fixed by the harness (so both options and all granularities build the same
thing); the plan step fills in the design: function signatures, edge cases, test cases.
"""

from dataclasses import dataclass

PKG = "orders_dedupe"
LAYOUT = [f"{PKG}/__init__.py", f"{PKG}/__main__.py", f"{PKG}/models.py", f"{PKG}/core.py",
          f"{PKG}/cli.py", "tests/test_core.py", "tests/test_cli.py", "README.md"]
ALL_PACKAGE_FILES = "*"  # in needs_full: every existing package file


@dataclass(frozen=True)
class StepSpec:
    key: str
    step_type: str  # plan | codegen | select_tests | fix_errors
    summary: str  # one line; this is what Jev reads
    brief: str  # the instruction the model reads
    writes: tuple[str, ...] = ()
    needs_full: tuple[str, ...] = ()  # existing files passed in full; other package files as signatures
    plan_keys: tuple[str, ...] = ()  # which parts of the plan this step needs
    lines: int = 0  # rough lines of output expected; a router signal
    may_edit_package: bool = False  # cleanup may rewrite any package file (public API must survive)


PLAN = StepSpec(
    "plan", "plan", "Design the module APIs, edge cases and test cases for the CSV duplicate flagger",
    "Design the solution. Return ONE JSON object (no prose) with keys: "
    '"modules" (an object with one entry per path in <project_layout>, each {"purpose": str, "api": '
    '[Python signatures, one per string]}), "edge_cases" (list of strings), "test_cases" (list of '
    'strings), "cli_contract" (string: argv, exit codes, stderr messages, output columns).',
    lines=60,
)

FINE = [
    StepSpec("scaffold", "codegen", "Create package skeleton files",
             f"Create the package skeleton. {PKG}/__init__.py holds only a docstring and "
             f'__version__ = "0.1.0". {PKG}/__main__.py is a placeholder that imports nothing from the '
             "package and exits 0 (it is wired to the CLI in a later step).",
             writes=(f"{PKG}/__init__.py", f"{PKG}/__main__.py"), lines=15),
    StepSpec("models", "codegen", "Define the data models",
             "Implement the data models listed in the plan for models.py. No I/O.",
             writes=(f"{PKG}/models.py",), plan_keys=(), lines=40),
    StepSpec("core", "codegen", "Implement the core duplicate-detection logic",
             "Implement the core duplicate-detection function(s) listed in the plan for core.py, using the "
             "models. Implement both matching rules; leave unusual-input handling to the next step.",
             writes=(f"{PKG}/core.py",), needs_full=(f"{PKG}/models.py",), lines=80),
    StepSpec("edge_cases", "codegen", "Harden core logic for edge cases",
             "Harden core.py for every item in edge_cases (whitespace, case, integer quantity "
             "normalization, empty input, chains of duplicates). Rewrite core.py in full; keep every "
             "public name it already defines.",
             writes=(f"{PKG}/core.py",), needs_full=(f"{PKG}/core.py", f"{PKG}/models.py"),
             plan_keys=("edge_cases",), lines=50),
    StepSpec("cli", "codegen", "Wire CSV input/output and the command-line interface",
             "Implement cli.py (argument parsing, CSV reading and writing, error handling, exit codes, "
             "summary line) per cli_contract, using the models and core. Rewrite __main__.py so that "
             "`python -m orders_dedupe` runs cli.main().",
             writes=(f"{PKG}/cli.py", f"{PKG}/__main__.py"), needs_full=(f"{PKG}/models.py",),
             plan_keys=("cli_contract",), lines=70),
    StepSpec("unit_tests", "codegen", "Write unit tests for core logic and CLI",
             "Write pytest tests: tests/test_core.py for the core logic, tests/test_cli.py for the CLI "
             "(run it with subprocess and `python -m orders_dedupe`). Cover every item in test_cases.",
             writes=("tests/test_core.py", "tests/test_cli.py"), plan_keys=("test_cases", "cli_contract"),
             lines=150),
    StepSpec("docs_cleanup", "codegen", "Write README and tidy code",
             "Write README.md (usage, examples, duplicate rules). You may also rewrite package files to "
             "remove dead code or improve docstrings, but behaviour and every public name must not change.",
             writes=("README.md",), needs_full=(ALL_PACKAGE_FILES,), may_edit_package=True, lines=60),
]


def _merge(specs: list[StepSpec]) -> StepSpec:
    uniq = lambda xs: tuple(dict.fromkeys(x for s in specs for x in xs(s)))  # noqa: E731
    return StepSpec(
        key="_".join(s.key for s in specs), step_type="codegen",
        summary=" + ".join(s.summary for s in specs),
        brief="\n".join(f"Part {i}: {s.brief}" for i, s in enumerate(specs, 1)),
        writes=uniq(lambda s: s.writes), needs_full=uniq(lambda s: s.needs_full),
        plan_keys=uniq(lambda s: s.plan_keys), lines=sum(s.lines for s in specs),
        may_edit_package=any(s.may_edit_package for s in specs),
    )


COARSE = [_merge(FINE[0:2]), _merge(FINE[2:4]), _merge(FINE[4:6]), FINE[6]]


def codegen_steps(granularity: str) -> list[StepSpec]:
    return {"fine": FINE, "coarse": COARSE}[granularity]


# The two steps whose inputs are produced at run time (so they have no static writes/needs).
SELECT_TESTS = StepSpec(
    "select_tests", "select_tests", "Choose which test files to run first",
    "Choose which test files to run first to get the fastest useful feedback on the files that "
    'changed. Return ONE JSON object (no prose): {"targets": ["tests/test_x.py", ...]}, using only '
    "paths from <candidates>.", lines=3)
FIX_ERRORS = StepSpec(
    "fix_errors", "fix_errors", "Fix failing tests or errors",
    "The tests below fail. Fix the root cause with the smallest change; edit the code or the test, "
    "whichever is actually wrong. Return every file you change in full.", lines=30)
```

The sample task the agent is given (unchanged from the earlier harness):

```markdown
# Task: `orders_dedupe` CLI

Build a Python 3.11 command-line tool that reads a CSV of orders and flags duplicate orders.
Runtime code uses the standard library only. Write unit tests with pytest in `tests/`.

## Layout

- The package lives in `orders_dedupe/` at the project root (not under `src/`), so that
  `python -m orders_dedupe` works when run from the project root.
- Tests live in `tests/` and must pass with `python -m pytest` from the project root.

## Input

A UTF-8 CSV file with a header row. Required columns: `order_id`, `customer_email`, `sku`,
`quantity`, `order_date`. Any other columns are allowed and must be preserved.

## Duplicate rules

Process rows in file order. A row is a duplicate of the earliest previous row it matches, where
two rows match if either:

1. their `order_id` values are equal after stripping surrounding whitespace; or
2. all of the following are equal:
   - `customer_email`, compared case-insensitively after stripping whitespace
   - `sku`, compared case-insensitively after stripping whitespace
   - `quantity`, compared as integers (`"2"`, `" 2 "` and `"02"` are equal)
   - `order_date`, compared after stripping whitespace

The first occurrence is never a duplicate.

## CLI

```
python -m orders_dedupe INPUT_CSV [--output OUTPUT_CSV]
```

- Writes a CSV (to stdout, or to `OUTPUT_CSV` if given) containing every input row in the
  original order, with all original columns in their original order, followed by two new columns:
  - `is_duplicate`: `true` or `false`
  - `duplicate_of`: the `order_id` (stripped) of the row it duplicates, or empty
- Prints a summary line to stderr exactly in the form `N rows, M duplicates`.
- Exit code 0 on success.
- If a required column is missing: print an error to stderr containing `missing column` and the
  column name, and exit with code 2.
- If the input file does not exist: print an error to stderr and exit with code 2.
- A file with only a header row is valid: output the header only, summary `0 rows, 0 duplicates`.
```

## 6. Output checks: what counts as a failure

A step's output has to pass these checks, in order. A failure is what triggers escalation (Step 8):

1. **Malformed response:** no `<file>` blocks, files outside the step's allowed set, a required file missing, bad JSON for plan/test selection, or `stop_reason` of `max_tokens` or `refusal`.
2. **Code doesn't run:** a syntax error, a module that fails to import, `python -m orders_dedupe --help` failing, or new tests that don't even collect.
3. **Inconsistent with earlier sub-steps:** a rewritten file dropped a public name an earlier step defined, or any file imports a module or name that doesn't exist (`check_internal_imports`). This is how a later sub-step contradicting an earlier one gets caught immediately, instead of surfacing as a confusing test failure.
4. **Tests fail:** this is checked in the `fix_errors` step: after the model's edit, the failing test run must pass or that attempt counts as failed. (Failing tests are not held against the `unit_tests` step itself, because they usually point at a bug in earlier code. They get their own fix step, as you specified.)

A failed attempt restores the files it touched, so the retry starts from a clean state.

```python
"""Parsing and validation of model output. A failed check is what triggers escalation."""

import ast
import json
import re
import sys

from .workspace import Workspace

PKG = "orders_dedupe"
FILE_RE = re.compile(r'<file path="([^"]+)">\n?(.*?)\n?</file>', re.S)


def parse_files(text: str) -> dict[str, str]:
    return {p.strip(): body + "\n" for p, body in FILE_RE.findall(text)}


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        v = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) else None


def safe_path(p: str) -> bool:
    return not p.startswith("/") and ".." not in p.split("/")


# --- AST helpers: top-level names, API digests ---------------------------------------------

def top_level_names(src: str, include_private: bool = True) -> set[str]:
    names: set[str] = set()
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return names if include_private else {x for x in names if not x.startswith("_")}


def _sig(n: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    ret = f" -> {ast.unparse(n.returns)}" if n.returns else ""
    return f"def {n.name}({ast.unparse(n.args)}){ret}"


def digest(src: str) -> str:
    """Signatures only: what a later step needs to stay consistent, at a fraction of the tokens."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "(unparseable)"
    out: list[str] = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_sig(n))
        elif isinstance(n, ast.ClassDef):
            out += [f"@{ast.unparse(d)}" for d in n.decorator_list]
            out.append(f"class {n.name}:")
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append("    " + _sig(m))
                elif isinstance(m, ast.AnnAssign):
                    out.append("    " + ast.unparse(m))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and len(ast.unparse(n)) < 120:
            out.append(ast.unparse(n))
    return "\n".join(out) or "(no public definitions)"


def error_type(output: str) -> str:
    for pat, name in [(r"ModuleNotFoundError|ImportError", "ImportError"), (r"SyntaxError", "SyntaxError"),
                      (r"AttributeError", "AttributeError"), (r"TypeError", "TypeError"),
                      (r"AssertionError|\bassert ", "AssertionError"), (r"timed out", "Timeout")]:
        if re.search(pat, output):
            return name
    return "Other"


# --- Validation ------------------------------------------------------------------------------

def check_internal_imports(ws: Workspace) -> str | None:
    """Every `from orders_dedupe.x import y` must resolve to a module and name that exist now.

    This is the check that catches a later sub-step contradicting an earlier one.
    """
    files = ws.files()
    for path in files:
        if not path.endswith(".py") or not (path.startswith(PKG + "/") or path.startswith("tests/")):
            continue
        try:
            tree = ast.parse(ws.read(path))
        except SyntaxError as e:
            return f"{path}: SyntaxError: {e}"
        for n in ast.walk(tree):
            if not isinstance(n, ast.ImportFrom):
                continue
            mod = PKG + (f".{n.module}" if n.module else "") if n.level else (n.module or "")
            if mod != PKG and not mod.startswith(PKG + "."):
                continue
            target = mod.replace(".", "/")
            src_path = next((c for c in (f"{target}.py", f"{target}/__init__.py") if c in files), None)
            if src_path is None:
                return f"{path}: imports module {mod}, which does not exist"
            names = top_level_names(ws.read(src_path))
            for a in n.names:
                if a.name != "*" and a.name not in names and f"{target}/{a.name}.py" not in files:
                    return f"{path}: imports {a.name} from {mod}, which does not define it"
    return None


def validate(ws: Workspace, new: dict[str, str], before: dict[str, str | None]) -> str | None:
    """Run after the files are written. Returns a failure reason, or None if all checks pass."""
    for p, src in new.items():  # 1. syntax
        if p.endswith(".py"):
            try:
                ast.parse(src)
            except SyntaxError as e:
                return f"{p}: SyntaxError: {e.msg} (line {e.lineno})"
    for p, src in new.items():  # 2. a rewritten file must keep the public names earlier steps relied on
        old = before.get(p)
        if old and p.startswith(PKG + "/") and p.endswith(".py"):
            lost = top_level_names(old, False) - top_level_names(src, False)
            if lost:
                return f"{p}: dropped public names {sorted(lost)} that earlier steps defined"
    bad = check_internal_imports(ws)  # 3. cross-file consistency
    if bad:
        return f"inconsistent with earlier code: {bad}"
    for p in new:  # 4. the code actually imports
        if p.startswith(PKG + "/") and p.endswith(".py") and not p.endswith("__main__.py"):
            mod = p[:-3].replace("/", ".").removesuffix(".__init__")
            r = ws.run([sys.executable, "-c", f"import {mod}"])
            if r.returncode:
                return f"import {mod} failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}"
    if f"{PKG}/__main__.py" in new:  # 5. the entry point starts
        r = ws.run([sys.executable, "-m", PKG, "--help"])
        if r.returncode:
            return f"python -m {PKG} --help exited {r.returncode}: {r.stderr.strip()[-200:]}"
    tests = [p for p in new if p.startswith("tests/") and p.endswith(".py")]
    if tests:  # 6. new tests are collectable (whether they PASS is the fix step's business)
        r = ws.pytest(["--collect-only", *tests])
        if not r.ok:
            return f"tests do not collect: {r.output.strip()[-300:]}"
    return None
```

```python
"""The agent's project directory. Reuses the earlier harness's Workspace and adds snapshots."""

import os
import subprocess

from harness.workspace import TestRun, Workspace as _Workspace  # noqa: F401  (TestRun re-exported)


# Files are rewritten in place within the same second. Python invalidates a .pyc by (mtime in whole
# seconds, size), so a rewrite of equal size can silently run stale bytecode. Never write any.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


class Workspace(_Workspace):
    def snapshot(self, paths) -> dict[str, str | None]:
        return {p: (self.read(p) if self._path(p).exists() else None) for p in paths}

    def restore(self, snap: dict[str, str | None]) -> None:
        for p, content in snap.items():
            if content is None:
                self._path(p).unlink(missing_ok=True)
            else:
                self.write(p, content)

    def run(self, cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, 124, "", "timed out")

    def test_counts(self) -> dict[str, int]:
        """Collected test count per test file, e.g. {"tests/test_core.py": 9}."""
        counts: dict[str, int] = {}
        for line in self.pytest(["--collect-only"]).output.splitlines():
            if "::" in line:
                f = line.split("::")[0].strip()
                counts[f] = counts.get(f, 0) + 1
        return counts


__all__ = ["Workspace", "TestRun"]
```

> **A bug the dry run found:** Python decides whether a `.pyc` is stale from the source file's mtime in whole seconds plus its size. The agent rewrites files in place within the same second, so a same-size rewrite can silently execute stale bytecode. My stub's placeholder and final `__main__.py` happened to be exactly 52 bytes. `workspace.py` therefore sets `PYTHONDONTWRITEBYTECODE=1` for every subprocess. Keep that line.

## 7. The routers: Option 1 (fixed) and Option 2 (Jev)

**(a) How the harness describes a step to Jev.** `JevRouter.route` sends one JSON `state`: the step key and type, a one-line summary, and signals the agent loop computed: `files_to_write`, `lines_expected`, `files_it_depends_on`, `context_tokens_est` (size of the prompt about to be sent), `error_type` (for fix and test-selection steps, from the latest pytest output), `retry_count` (fix rounds so far), and `candidate_test_files` for test selection. The three options, Haiku, Sonnet, and Opus, are fixed in `OPTIONS` with a one-line description each. Jev picks among them and does no coding.

**(b) Choice, not Score.** I used a single **Choice** question over the three models:
- Jev's answer maps one-to-one onto the options the harness prepared, so there is nothing to translate.
- Its confidence attaches to *the pick itself*, which is exactly what the gate in (c) needs.
- Score would add a second step: rate complexity, then map score ranges to tiers with cutpoints you'd have to invent. That's one more hyperparameter, and the confidence would be about the score bucket, not the model decision.

Switch to Score later if you want to *learn* the mapping: log each step's score alongside whether the cheap model's first attempt passed (the ledger already records it), fit the cutpoints from that, and compare against Choice.

**(c) The confidence gate.** If Jev's confidence is below `jev_min_confidence` (default 0.75, set with `--min-confidence`) and the pick is below Opus, the step moves up: one tier (`low_conf_policy="one_tier"`) or straight to Opus (`"opus"`). A low-confidence Opus pick stays Opus. The default threshold is a guess; tune it from your logs (Step 11 shows how often it fires).

```python
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
```

## 8. The agent loop (identical for both options)

`Agent.run_step` is the whole mechanism. Both options execute exactly this code, so cost differences come from model choice and nothing else:

1. Build the prompt (shared task + layout prefix, then a step-specific body) and compute `context_tokens_est`.
2. **Route once.** Option 1's router returns Opus. Option 2's calls Jev and applies the gate.
3. Call the chosen model; run the step's check (Step 6).
4. **Escalation rule (d), fixed in code and not a Jev call:** on failure in Option 2, retry the step one tier higher (Haiku to Sonnet to Opus) and log `"haiku->sonnet: <reason>"` plus the money spent on the failed attempt. At Opus (and always in Option 1, which has nothing above Opus) the retry stays on the same model. Both options get the same `max_attempts_per_step`.
5. **Logging (e):** the `StepRecord` stores Jev's pick, its confidence, whether the gate bumped it, the first tier run, and the tier whose output was finally accepted.

The surrounding loop is `plan` then each code-generation step, then `select_tests -> run -> fix_errors` rounds. After the selected tests pass, the full suite runs as the gate.

```python
"""The shared agent loop. Option 1 and Option 2 run this exact code; only the router differs."""

import json
import re

from . import checks
from .clients import Claude
from .config import Settings, TIERS
from .ledger import Ledger, StepRecord
from .router import Route
from .steps import (ALL_PACKAGE_FILES, FIX_ERRORS, LAYOUT, PKG, PLAN, SELECT_TESTS, StepSpec,
                    codegen_steps)
from .workspace import Workspace

FILES_FORMAT = ('Return every file you write in full, each as <file path="PATH">...</file>, and nothing '
                "else outside those blocks.")
JSON_PLAN_KEYS = ("modules", "edge_cases", "test_cases", "cli_contract")


class StepFailed(RuntimeError):
    pass


class Agent:
    def __init__(self, cfg: Settings, option: int, ws: Workspace, task: str, ledger: Ledger,
                 claude: Claude, router):
        self.cfg, self.option, self.ws, self.ledger = cfg, option, ws, ledger
        self.claude, self.router = claude, router
        self.prefix = f"<task>\n{task}\n</task>\n<project_layout>\n" + "\n".join(LAYOUT) + "\n</project_layout>"
        self.plan: dict = {}
        self.dirty: set[str] = set()  # files changed since tests last ran
        self.fix_rounds = 0
        self.last_error = ""

    # ---- generic step runner: route once, then attempt / validate / escalate ----------------
    def blocks(self, body: str) -> list[dict]:
        shared = {"type": "text", "text": self.prefix}
        if self.cfg.cache:
            shared["cache_control"] = {"type": "ephemeral"}
        return [shared, {"type": "text", "text": body}]

    def run_step(self, spec: StepSpec, body: str, handle, signals: dict) -> None:
        blocks = self.blocks(body)
        signals = {**signals, "context_tokens_est": sum(len(b["text"]) for b in blocks) // 4}
        route: Route = self.router.route(spec, signals)  # the ONLY routing call for this step
        rec = StepRecord(step=spec.key, step_type=spec.step_type, signals=signals, jev_choice=route.jev_choice,
                         jev_confidence=route.jev_confidence, first_tier=route.tier,
                         low_confidence_bump=route.bumped)
        self.ledger.steps.append(rec)
        idx = TIERS.index(route.tier)
        for attempt in range(1, self.cfg.max_attempts_per_step + 1):
            tier = TIERS[idx]
            rec.attempts = attempt
            text, stop, call = self.claude.call(tier, blocks, step=spec.key, step_type=spec.step_type,
                                                attempt=attempt)
            if stop in ("max_tokens", "refusal"):
                ok, reason = False, f"malformed: stop_reason={stop}"
            else:
                ok, reason = handle(text)
            call.ok, call.note = ok, reason[:200]
            if ok:
                rec.ok, rec.final_tier = True, tier
                return
            if self.option == 2 and idx < len(TIERS) - 1:  # (d) fixed escalation rule: one tier up
                rec.escalations.append(f"{tier}->{TIERS[idx + 1]}: {reason[:120]}")
                rec.escalation_cost_usd += call.cost_usd
                idx += 1
            else:
                rec.same_tier_retries += attempt < self.cfg.max_attempts_per_step
        raise StepFailed(f"{spec.key} failed {self.cfg.max_attempts_per_step} attempts: {reason[:200]}")

    # ---- context: pass forward only what the next step needs --------------------------------
    def package_files(self) -> list[str]:
        return [p for p in self.ws.files() if p.startswith(PKG + "/") and p.endswith(".py")]

    def code_context(self, spec: StepSpec) -> str:
        existing = self.package_files()
        full = existing if ALL_PACKAGE_FILES in spec.needs_full else [p for p in spec.needs_full if p in existing]
        parts = [f'<file path="{p}">\n{self.ws.read(p)}</file>' for p in full]
        sigs = [f"# {p}\n{checks.digest(self.ws.read(p))}" for p in existing if p not in full]
        if sigs:
            parts.append("<api_of_other_files>\n" + "\n\n".join(sigs) + "\n</api_of_other_files>")
        return "\n".join(parts)

    def plan_slice(self, spec: StepSpec) -> str:
        mods = self.plan["modules"]
        paths = set(spec.writes) | (set(mods) if ALL_PACKAGE_FILES in spec.needs_full else set(spec.needs_full))
        piece = {"modules": {p: mods[p] for p in LAYOUT if p in paths}}
        piece.update({k: self.plan[k] for k in spec.plan_keys})
        return json.dumps(piece, indent=1)

    # ---- steps ------------------------------------------------------------------------------
    def step_plan(self) -> None:
        def handle(text: str):
            plan = checks.parse_json(text)
            if not plan or any(k not in plan for k in JSON_PLAN_KEYS):
                return False, f"malformed: plan JSON missing one of {JSON_PLAN_KEYS}"
            if any(p not in plan["modules"] for p in LAYOUT):
                return False, "malformed: plan.modules lacks a path from the layout"
            self.plan = plan
            return True, ""
        self.run_step(PLAN, PLAN.brief, handle,
                      {"files_to_write": 0, "lines_expected": PLAN.lines, "error_type": None, "retry_count": 0})

    def apply_files(self, text: str, allowed, required=()) -> tuple[bool, str]:
        files = checks.parse_files(text)
        if not files:
            return False, "malformed: no <file> blocks"
        bad = [p for p in files if not checks.safe_path(p) or not allowed(p)]
        if bad:
            return False, f"malformed: files outside the allowed set: {bad}"
        if any(p not in files for p in required):
            return False, f"malformed: missing required files {[p for p in required if p not in files]}"
        before = self.ws.snapshot(files)
        for p, content in files.items():
            self.ws.write(p, content)
        reason = checks.validate(self.ws, files, before)
        if reason:
            self.ws.restore(before)  # a failed attempt leaves no trace; the retry starts clean
            return False, reason
        self.dirty |= set(files)
        return True, ""

    def step_codegen(self, spec: StepSpec) -> None:
        def allowed(p: str) -> bool:
            return p in spec.writes or (spec.may_edit_package and p.startswith(PKG + "/") and p.endswith(".py"))
        body = (f"<plan>\n{self.plan_slice(spec)}\n</plan>\n<existing_code>\n{self.code_context(spec)}\n"
                f"</existing_code>\n<step>\n{spec.brief}\n</step>\n"
                f"Write exactly these files: {', '.join(spec.writes)}\n{FILES_FORMAT}")
        existing = [p for p in spec.needs_full if p in self.package_files()]
        self.run_step(spec, body, lambda t: self.apply_files(t, allowed, spec.writes),
                      {"files_to_write": len(spec.writes), "lines_expected": spec.lines,
                       "files_it_depends_on": len(existing), "error_type": None, "retry_count": 0})

    def step_select_tests(self, rnd: int) -> list[str]:
        counts = self.ws.test_counts()
        if not counts:
            return []  # nothing to choose from: the full run will report "no tests"
        body = ("<candidates>\n" + "\n".join(f"{p} ({n} tests)" for p, n in counts.items()) +
                f"\n</candidates>\n<changed_files>\n{', '.join(sorted(self.dirty)) or '(none)'}\n</changed_files>\n"
                f"<step>\n{SELECT_TESTS.brief}\n</step>")
        chosen: list[str] = []

        def handle(text: str):
            data = checks.parse_json(text)
            t = data.get("targets") if data else None
            if not isinstance(t, list) or not t or any(x not in counts for x in t):
                return False, "malformed: targets must be a non-empty list drawn from the candidates"
            chosen[:] = t
            return True, ""
        self.run_step(SELECT_TESTS, body, handle,
                      {"files_to_write": 0, "lines_expected": SELECT_TESTS.lines, "candidate_test_files": len(counts),
                       "error_type": checks.error_type(self.last_error) if self.last_error else None,
                       "retry_count": rnd})
        return chosen

    def step_fix(self, output: str, targets: list[str]) -> None:
        implicated = [p for p in dict.fromkeys(re.findall(rf"({PKG}/\w+\.py|tests/\w+\.py)", output))
                      if p in self.ws.files()]
        implicated = implicated or self.package_files() + self.ws.test_files()
        src = "\n".join(f'<file path="{p}">\n{self.ws.read(p)}</file>' for p in implicated)
        body = (f"<failure>\n{output[-5000:]}\n</failure>\n<files>\n{src}\n</files>\n"
                f"<step>\n{FIX_ERRORS.brief}\n</step>\n{FILES_FORMAT}")

        def allowed(p: str) -> bool:
            return (p.startswith(PKG + "/") or p.startswith("tests/")) and p.endswith(".py")

        def handle(text: str):
            ok, reason = self.apply_files(text, allowed)
            if not ok:
                return ok, reason
            res = self.ws.pytest(targets or None)  # the fix must actually turn the failing run green
            if not res.ok:
                return False, f"tests still failing: {res.output.strip()[-150:]}"
            return True, ""
        self.run_step(FIX_ERRORS, body, handle,
                      {"files_to_write": len(implicated), "lines_expected": FIX_ERRORS.lines,
                       "error_type": checks.error_type(output), "retry_count": self.fix_rounds})

    def test_and_fix(self) -> None:
        for rnd in range(self.cfg.max_fix_rounds + 1):
            targets = self.step_select_tests(rnd)
            res, ran = self.ws.pytest(targets or None), targets
            if res.ok and targets:  # the selected tests pass: run the full suite as the gate
                res, ran = self.ws.pytest(), []
            self.dirty.clear()
            if res.ok:
                return
            self.last_error = res.output
            if rnd == self.cfg.max_fix_rounds:
                raise StepFailed("tests still failing after the maximum number of fix rounds")
            self.step_fix(res.output, ran)
            self.fix_rounds += 1

    def run(self) -> None:
        self.step_plan()
        for spec in codegen_steps(self.cfg.granularity):
            self.step_codegen(spec)
        self.test_and_fix()
```

## 9. Run matrix: four configurations, N repeats

`runner.py` runs `{Option 1, Option 2} x {fine, coarse}`, `--runs` times each (default 3). Configurations are **interleaved within each repeat** (not all of one config, then the next), so API slowdowns and rate-limit weather affect every configuration equally. Each run gets a fresh workspace and ledger, and the full data is saved to `all_runs.json` so you can regenerate the report any time with `python -m routing.report <path>`.

```python
"""Runs the matrix: {Option 1, Option 2} x {fine, coarse} x N repeats, then writes the report."""

import argparse
import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from harness import quality

from .agent import Agent, StepFailed
from .clients import Claude, Jev
from .config import Settings
from .ledger import Ledger
from .report import render
from .router import FixedRouter, JevRouter
from .workspace import Workspace

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [(1, "fine"), (1, "coarse"), (2, "fine"), (2, "coarse")]


def run_once(option: int, granularity: str, run: int, cfg: Settings, out_dir: Path) -> dict:
    label = f"opt{option}-{granularity}"
    ws_root = out_dir / label / f"run{run}" / "workspace"
    shutil.copytree(ROOT / "task" / "starter", ws_root)
    ledger = Ledger()
    cfg = Settings(**{**asdict(cfg), "granularity": granularity})
    router = FixedRouter() if option == 1 else JevRouter(Jev(ledger), cfg)
    agent = Agent(cfg, option, Workspace(ws_root), (ROOT / "task" / "TASK.md").read_text(), ledger,
                  Claude(ledger), router)
    print(f"== {label} run {run} ==", flush=True)
    t0, error = time.perf_counter(), ""
    try:
        agent.run()
    except StepFailed as e:
        error = str(e)
    wall = time.perf_counter() - t0
    q = quality.check(ws_root)  # held-out acceptance tests the agent never saw: same gate for every config
    result = {"config": label, "option": option, "granularity": granularity, "run": run,
              "success": q.succeeded and not error, "error": error, "wall_s": wall,
              "own_tests": [q.own_passed, q.own_total], "acceptance": [q.acceptance_passed, q.acceptance_total],
              **ledger.to_dict()}
    (ws_root.parent / "result.json").write_text(json.dumps(result, indent=1))
    cost = sum(c["cost_usd"] for c in result["calls"])
    print(f"   success={result['success']} acceptance={q.acceptance_passed}/{q.acceptance_total} "
          f"cost=${cost:.4f} {error}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=3, help="repeat runs per configuration (default 3)")
    ap.add_argument("--granularity", choices=["fine", "coarse", "both"], default="both",
                    help="code-generation split: 7 sub-steps, 4 groups, or run both (default)")
    ap.add_argument("--configs", default="all", help="e.g. 1fine,2coarse (default: all four)")
    ap.add_argument("--min-confidence", type=float, default=Settings.jev_min_confidence)
    ap.add_argument("--low-conf-policy", choices=["one_tier", "opus"], default=Settings.low_conf_policy)
    ap.add_argument("--no-cache", action="store_true", help="drop cache_control from every request")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = Settings(jev_min_confidence=args.min_confidence, low_conf_policy=args.low_conf_policy,
                   cache=not args.no_cache)
    wanted = CONFIGS if args.configs == "all" else [(int(c[0]), c[1:]) for c in args.configs.split(",")]
    wanted = [c for c in wanted if args.granularity in ("both", c[1])]
    out_dir = args.out or ROOT / "runs" / datetime.now().strftime("routing-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for run in range(1, args.runs + 1):  # interleave configs so drift (load, prices) hits all equally
        for option, gran in wanted:
            results.append(run_once(option, gran, run, cfg, out_dir))
            (out_dir / "all_runs.json").write_text(json.dumps(results, indent=1))
    report = render(results, asdict(cfg))
    (out_dir / "report.md").write_text(report)
    print("\n" + report + f"\n\nSaved to {out_dir}")
```

```python
# run_routing.py
"""python run_routing.py [--runs 3] [--configs 1fine,2coarse] [--min-confidence 0.75]"""
from routing.runner import main

main()
```

## 10. Quality check: is the cost comparison fair?

After every run, the harness runs two test sets **against the generated code, outside the agent's view**: the agent's own tests (counted, not trusted) and the 11 held-out acceptance tests in `task/acceptance/`, via `harness.quality.check`. A run counts as a success only if all acceptance tests pass and no step failed outright. The report prints success per run and also computes **cost over successful runs only**, so a cheap run that produced broken code can't make a configuration look better. Steps where a cheaper model's output needed escalation are recorded per step (`escalations`), and their wasted spend is a column in the report. Those retries are part of Option 2's true cost and they are already inside its totals.

## 11. The report

Everything is computed from the ledgers: per-config cost and latency (mean, min, max), cost per step type and per code-generation sub-step/group, calls per model including Jev, routing distribution by step (Jev's raw pick, the model that ran first after the gate, and the model whose output was accepted), escalations and what they cost, Jev's share of total cost, success per run, and Option 2 vs Option 1 at each granularity with the winner.

```python
"""Markdown comparison report from the per-run ledgers. Re-run on saved data with:

    python -m routing.report runs/<dir>/all_runs.json
"""

import json
import sys
from collections import Counter
from statistics import mean

from .config import PRICES, TIERS

CONFIG_ORDER = ["opt1-fine", "opt1-coarse", "opt2-fine", "opt2-coarse"]
NAMES = {"opt1-fine": "Opt 1 fine", "opt1-coarse": "Opt 1 coarse", "opt2-fine": "Opt 2 fine", "opt2-coarse": "Opt 2 coarse"}


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def usd(x: float) -> str:
    sign, x = ("-" if x < 0 else ""), abs(x)
    return f"{sign}${x:.4f}" if x >= 0.001 or x == 0 else f"{sign}${x:.6f}"


pct = lambda x: f"{x:+.1f}%"  # noqa: E731


def mmm(xs: list[float], f=usd) -> str:
    return f"{f(mean(xs))} ({f(min(xs))} to {f(max(xs))})" if xs else "n/a"


def cost(r: dict, **where) -> float:
    return sum(c["cost_usd"] for c in r["calls"] if all(c[k] == v for k, v in where.items()))


def per_run_mean(runs: list[dict], fn) -> float:
    return mean(fn(r) for r in runs) if runs else 0.0


def render(results: list[dict], settings: dict | None = None) -> str:
    by_cfg = {c: [r for r in results if r["config"] == c] for c in CONFIG_ORDER}
    by_cfg = {c: rs for c, rs in by_cfg.items() if rs}
    md: list[str] = ["# Routing experiment report\n"]

    md.append("## Prices used\n")
    md.append(table(["Model", "Input $/MTok", "Output $/MTok", "Cache write 5m", "Cache read", "Checked", "Source"],
                    [[k, p.input, p.output, p.cache_write_5m, p.cache_read, p.checked, p.source] for k, p in PRICES.items()]))
    if settings:
        md.append(f"\nSettings: {settings}\n")

    # --- headline ---------------------------------------------------------------------------
    md.append("\n## Total cost and latency per configuration (mean, min to max across runs)\n")
    rows = []
    for c, rs in by_cfg.items():
        ok = [r for r in rs if r["success"]]
        rows.append([NAMES[c], len(rs), f"{len(ok)}/{len(rs)}", mmm([cost(r) for r in rs]),
                     usd(mean(cost(r) for r in ok)) if ok else "n/a",
                     mmm([sum(x["latency_s"] for x in r["calls"]) for r in rs], lambda v: f"{v:.1f}s"),
                     mmm([r["wall_s"] for r in rs], lambda v: f"{v:.1f}s")])
    md.append(table(["Config", "Runs", "Succeeded", "Total cost", "Cost, successful runs only",
                     "Sum of API latency", "Wall clock"], rows))

    # --- tokens / calls / latency per model ---------------------------------------------------
    md.append("\n## Calls, tokens and latency per model (mean per run)\n")
    rows = []
    for c, rs in by_cfg.items():
        for t in TIERS + ["jev"]:
            f = lambda k, t=t, rs=rs: per_run_mean(rs, lambda r: sum(x[k] for x in r["calls"] if x["tier"] == t))  # noqa: E731
            n = per_run_mean(rs, lambda r, t=t: sum(1 for x in r["calls"] if x["tier"] == t))
            if n:
                rows.append([NAMES[c], t, f"{n:.1f}", f"{f('input_tokens'):.0f}", f"{f('output_tokens'):.0f}",
                             f"{f('cache_read_tokens'):.0f}/{f('cache_write_tokens'):.0f}", f"{f('latency_s'):.1f}s",
                             usd(f("cost_usd"))])
    md.append(table(["Config", "Model", "Calls", "Input tok", "Output tok", "Cache read/write tok", "Latency", "Cost"], rows))

    # --- cost per step ----------------------------------------------------------------------
    for gran in ("fine", "coarse"):
        o1, o2 = by_cfg.get(f"opt1-{gran}", []), by_cfg.get(f"opt2-{gran}", [])
        if not (o1 or o2):
            continue
        keys = list(dict.fromkeys(s["step"] for r in o1 + o2 for s in r["steps"]))
        md.append(f"\n## Cost per step, {gran} granularity (mean per run)\n")
        rows = []
        for k in keys:
            typ = next(s["step_type"] for r in o1 + o2 for s in r["steps"] if s["step"] == k)
            rows.append([f"{k} ({typ})", usd(per_run_mean(o1, lambda r: cost(r, step=k, kind="model"))),
                         usd(per_run_mean(o2, lambda r: cost(r, step=k, kind="model"))),
                         usd(per_run_mean(o2, lambda r: cost(r, step=k, kind="router")))])
        rows.append(["**Total**", usd(per_run_mean(o1, cost)), usd(per_run_mean(o2, lambda r: cost(r, kind="model"))),
                     usd(per_run_mean(o2, lambda r: cost(r, kind="router")))])
        md.append(table(["Step", "Opt 1 (Opus)", "Opt 2 Claude calls", "Opt 2 Jev routing"], rows))

    # --- routing distribution -----------------------------------------------------------------
    for gran in ("fine", "coarse"):
        rs = by_cfg.get(f"opt2-{gran}", [])
        if not rs:
            continue
        steps = [s for r in rs for s in r["steps"]]
        groups: dict[str, list[dict]] = {}
        for k in dict.fromkeys(s["step"] for s in steps):
            groups[k] = [s for s in steps if s["step"] == k]
        groups["codegen (all)"] = [s for s in steps if s["step_type"] == "codegen"]
        groups["ALL steps"] = steps

        def dist(ss: list[dict], field: str) -> str:
            c = Counter(s[field] for s in ss if s[field])
            n = sum(c.values()) or 1
            return " / ".join(f"{100 * c[t] // n}" for t in TIERS)

        md.append(f"\n## Routing distribution, Option 2 {gran} (% of steps: haiku / sonnet / opus)\n")
        md.append(table(["Step", "Steps", "Jev's raw pick", "Ran first (after confidence gate)", "Accepted output from"],
                        [[k, len(ss), dist(ss, "jev_choice"), dist(ss, "first_tier"), dist(ss, "final_tier")]
                         for k, ss in groups.items()]))

    # --- escalations and overhead -------------------------------------------------------------
    md.append("\n## Escalations, retries and Jev overhead (mean per run)\n")
    rows = []
    for c, rs in by_cfg.items():
        steps = lambda r: r["steps"]  # noqa: E731
        total = per_run_mean(rs, cost)
        jev = per_run_mean(rs, lambda r: cost(r, kind="router"))
        rows.append([NAMES[c],
                     f"{per_run_mean(rs, lambda r: sum(len(s['escalations']) for s in steps(r))):.2f}",
                     usd(per_run_mean(rs, lambda r: sum(s["escalation_cost_usd"] for s in steps(r)))),
                     f"{per_run_mean(rs, lambda r: sum(s['same_tier_retries'] for s in steps(r))):.2f}",
                     f"{per_run_mean(rs, lambda r: sum(s['low_confidence_bump'] for s in steps(r))):.2f}",
                     f"{per_run_mean(rs, lambda r: sum(1 for s in steps(r) if r['option'] == 2 and s['jev_choice'] is None)):.2f}",
                     usd(jev), f"{100 * jev / total:.1f}%" if total and jev else "-"])
    md.append(table(["Config", "Escalations", "Cost added by escalations*", "Same-model retries",
                     "Low-confidence bumps", "Jev errors", "Jev routing cost", "Jev share of total cost"], rows))
    md.append("\n*Spend on attempts that failed and were then escalated; the retry itself is in the totals. "
              "Option 1 has no higher tier, so its failures show up as same-model retries.")

    # --- success ------------------------------------------------------------------------------
    md.append("\n## Did each run produce working code? (held-out acceptance tests)\n")
    md.append(table(["Config", "Per run (acceptance passed/total, own tests passed/total)"],
                    [[NAMES[c], "; ".join(f"run{r['run']}: {'PASS' if r['success'] else 'FAIL'} "
                                          f"{r['acceptance'][0]}/{r['acceptance'][1]}, {r['own_tests'][0]}/{r['own_tests'][1]}"
                                          + (f" [{r['error'][:60]}]" if r["error"] else "") for r in rs)]
                     for c, rs in by_cfg.items()]))

    # --- savings ------------------------------------------------------------------------------
    md.append("\n## Option 2 vs Option 1 (negative = Option 2 is cheaper)\n")
    rows, totals = [], {}
    for gran in ("fine", "coarse"):
        a, b = by_cfg.get(f"opt1-{gran}"), by_cfg.get(f"opt2-{gran}")
        if not (a and b):
            continue
        ca, cb = mean(cost(r) for r in a), mean(cost(r) for r in b)
        totals[gran] = cb
        sa, sb = [cost(r) for r in a if r["success"]], [cost(r) for r in b if r["success"]]
        ok = f"{usd(mean(sb) - mean(sa))} ({pct(100 * (mean(sb) - mean(sa)) / mean(sa))})" if sa and sb else "n/a"
        rows.append([gran, usd(ca), usd(cb), usd(cb - ca), pct(100 * (cb - ca) / ca), ok])
    md.append(table(["Granularity", "Opt 1 mean", "Opt 2 mean", "Difference", "Difference %", "Difference, successful runs only"], rows))
    if len(totals) == 2:
        best = min(totals, key=totals.get)
        md.append(f"\n**Option 2 is cheaper at {best} granularity** ({usd(totals[best])} vs "
                  f"{usd(max(totals.values()))}) once Jev overhead and escalations are included.")
    return "\n".join(md)


if __name__ == "__main__":
    print(render(json.load(open(sys.argv[1]))))
```

## 12. Dry run first, then the real run

```bash
python -m routing.dry_run                     # stubbed APIs, no keys, no spend; numbers are FAKE
python run_routing.py --runs 1                # live, smallest meaningful run: one repeat of all four configs
python run_routing.py                         # live, the default 3 repeats per config
python run_routing.py --granularity coarse --runs 5    # only the two coarse configurations
python run_routing.py --configs 2fine,2coarse --min-confidence 0.85 --low-conf-policy opus
```

**First live measurement (one `opt1-fine` run, Opus only):** $0.76 and 291 s of wall time, 11/11 acceptance tests passed. About 87% of that was output tokens (32.8K of them, thinking included), and the biggest steps were unit_tests (26%), plan (18%), edge_cases (17%) and docs_cleanup (15%). The default 12 runs should therefore cost on the order of $5 to $10, but Option 2 costs are still unmeasured, so check `--runs 1` first.

## 13. Factors that can distort the comparison

- **Prompt caching is per model.** Option 1 keeps one model's cache warm across steps. Option 2 switches models, so each model builds its own. The harness puts `cache_control` on the shared prefix in both options, and in the first live run the ~911-token shared prefix was read from cache on every Opus call (`cache_read_tokens: 911`; the minimum cacheable length is model-specific, see the [prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). At that size the dollars are tiny (about $0.0002 per read), but cache state also carries over between runs within the 5-minute window, so whichever configuration runs first may pay a one-time write (about $0.005 on Opus) that later ones don't. The ledger records cache tokens either way. Opus 5.5 cache reads cost $0.20/MTok, the same as Sonnet 5.5's, so on a bigger task with high hit rates, Option 1's input side gets much cheaper than the list prices suggest. Re-test on a larger task with `--no-cache` as the control.
- **Different token counts for the same text.** Anthropic's pricing page says Claude 4.7 and later models use a newer tokenizer that produces roughly 30% more tokens for the same text; Haiku 4.5 uses the older one. Don't compare token counts across tiers, only dollars.
- **Different context sizes per model.** Steps get different prompts depending on their spec, and a retry on a stronger model re-sends the same prompt. Fine granularity re-sends the shared prefix, plan slice, and API digests more often; the ledger's input tokens per call show how much.
- **Escalation retries** cost a full failed attempt before the successful one. Cheaper models can also fail in ways the validators don't catch (the code runs, the logic is subtly wrong). Those surface later as extra `fix_errors` rounds, which look like ordinary steps in the table but are really hidden escalations. Compare the number of `fix_errors` and `select_tests` calls across configurations.
- **Jev's added latency per step.** Every step pays a routing round trip before the real call, in sequence. Fine granularity pays it more often (7 code-generation routing calls instead of 4). Third-party pages claim 70 to 500 ms per call; the harness measures the real value (sum of Jev latency in the per-model table).
- **Integration mismatches.** Different models write different files, and each interprets the plan slightly differently. The consistency checks catch missing names and imports, but not subtle behavioral mismatches (for example, one model treating quantity as a string and another as an int).
- **Nondeterminism.** Model output varies run to run, and Jev's routing can too, so the same config can route differently across repeats.
- **Small samples.** 3 repeats give you a range, not a significance test. Treat any difference smaller than the min-to-max spread as noise.
- **Effort and thinking differ by tier.** Haiku 4.5 has no `effort` setting. Opus 5.5 and Sonnet 5.5 run adaptive thinking. Output tokens (which include thinking) are 5x the input price, so a model that thinks more can erase a lower per-token price.
- **Single task.** One CSV-dedupe task says little about other workloads. Don't generalize from it.

---

## Example final report (FAKE: produced by the stubbed dry run)

> **These numbers are not real results.** They come from `python -m routing.dry_run`, where scripted fake models return a known-good solution and the token counts are made up. They show what the report looks like and that every column fills in. Tables shortened: the prices table and the coarse-granularity step and routing tables are omitted (the real report has them).

### Total cost and latency per configuration (mean, min to max across runs)

| Config | Runs | Succeeded | Total cost | Cost, successful runs only | Sum of API latency | Wall clock |
|---|---:|---:|---:|---:|---:|---:|
| Opt 1 fine | 3 | 3/3 | $0.0801 ($0.0801 to $0.0801) | $0.0801 | 0.7s (0.7s to 0.7s) | 2.5s (2.4s to 2.7s) |
| Opt 1 coarse | 3 | 3/3 | $0.0598 ($0.0598 to $0.0598) | $0.0598 | 0.5s (0.5s to 0.5s) | 2.3s (2.1s to 2.4s) |
| Opt 2 fine | 3 | 3/3 | $0.0539 ($0.0539 to $0.0539) | $0.0539 | 0.6s (0.6s to 0.6s) | 4.1s (3.8s to 4.4s) |
| Opt 2 coarse | 3 | 3/3 | $0.0359 ($0.0359 to $0.0359) | $0.0359 | 0.4s (0.4s to 0.4s) | 3.8s (3.7s to 4.0s) |

### Calls, tokens and latency per model (mean per run)

| Config | Model | Calls | Input tok | Output tok | Cache read/write tok | Latency | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Opt 1 fine | opus | 9.0 | 6074 | 2791 | 0/0 | 0.7s | $0.0801 |
| Opt 1 coarse | opus | 6.0 | 4344 | 2123 | 0/0 | 0.5s | $0.0598 |
| Opt 2 fine | haiku | 8.0 | 5404 | 1906 | 0/0 | 0.2s | $0.0149 |
| Opt 2 fine | sonnet | 4.0 | 2829 | 1432 | 0/0 | 0.2s | $0.0200 |
| Opt 2 fine | opus | 2.0 | 1211 | 699 | 0/0 | 0.2s | $0.0188 |
| Opt 2 fine | jev | 11.0 | 3426 | 0 | 0/0 | 0.1s | $0.000144 |
| Opt 2 coarse | haiku | 5.0 | 3594 | 1166 | 0/0 | 0.1s | $0.0094 |
| Opt 2 coarse | sonnet | 3.0 | 2269 | 1299 | 0/0 | 0.1s | $0.0175 |
| Opt 2 coarse | opus | 1.0 | 484 | 344 | 0/0 | 0.1s | $0.0088 |
| Opt 2 coarse | jev | 8.0 | 2528 | 0 | 0/0 | 0.1s | $0.000106 |

### Cost per step, fine granularity (mean per run)

| Step | Opt 1 (Opus) | Opt 2 Claude calls | Opt 2 Jev routing |
|---|---:|---:|---:|
| plan (plan) | $0.0088 | $0.0088 | $0.000013 |
| scaffold (codegen) | $0.0072 | $0.0018 | $0.000013 |
| models (codegen) | $0.0074 | $0.0018 | $0.000013 |
| core (codegen) | $0.0080 | $0.0056 | $0.000013 |
| edge_cases (codegen) | $0.0100 | $0.0150 | $0.000013 |
| cli (codegen) | $0.0135 | $0.0068 | $0.000013 |
| unit_tests (codegen) | $0.0094 | $0.0024 | $0.000013 |
| docs_cleanup (codegen) | $0.0094 | $0.0023 | $0.000013 |
| select_tests (select_tests) | $0.0064 | $0.0032 | $0.000026 |
| fix_errors (fix_errors) | $0.0000 | $0.0060 | $0.000013 |
| **Total** | $0.0801 | $0.0537 | $0.000144 |

### Routing distribution, Option 2 fine (% of steps: haiku / sonnet / opus)

| Step | Steps | Jev's raw pick | Ran first (after confidence gate) | Accepted output from |
|---|---:|---:|---:|---:|
| plan | 3 | 0 / 0 / 100 | 0 / 0 / 100 | 0 / 0 / 100 |
| scaffold | 3 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| models | 3 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| core | 3 | 100 / 0 / 0 | 100 / 0 / 0 | 0 / 100 / 0 |
| edge_cases | 3 | 100 / 0 / 0 | 0 / 100 / 0 | 0 / 0 / 100 |
| cli | 3 | 0 / 100 / 0 | 0 / 100 / 0 | 0 / 100 / 0 |
| unit_tests | 3 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| docs_cleanup | 3 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| select_tests | 6 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| fix_errors | 3 | 100 / 0 / 0 | 100 / 0 / 0 | 0 / 100 / 0 |
| codegen (all) | 21 | 85 / 14 / 0 | 71 / 28 / 0 | 57 / 28 / 14 |
| ALL steps | 33 | 81 / 9 / 9 | 72 / 18 / 9 | 54 / 27 / 18 |

### Escalations, retries and Jev overhead (mean per run)

| Config | Escalations | Cost added by escalations* | Same-model retries | Low-confidence bumps | Jev errors | Jev routing cost | Jev share of total cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Opt 1 fine | 0.00 | $0.0000 | 0.00 | 0.00 | 0.00 | $0.0000 | - |
| Opt 1 coarse | 0.00 | $0.0000 | 0.00 | 0.00 | 0.00 | $0.0000 | - |
| Opt 2 fine | 3.00 | $0.0084 | 0.00 | 1.00 | 0.00 | $0.000144 | 0.3% |
| Opt 2 coarse | 1.00 | $0.0018 | 0.00 | 0.00 | 0.00 | $0.000106 | 0.3% |

*Spend on attempts that failed and were then escalated; the retry itself is in the totals. Option 1 has no higher tier, so its failures show up as same-model retries.

### Did each run produce working code? (held-out acceptance tests)

| Config | Per run (acceptance passed/total, own tests passed/total) |
|---|---:|
| Opt 1 fine | run1: PASS 11/11, 2/2; run2: PASS 11/11, 2/2; run3: PASS 11/11, 2/2 |
| Opt 1 coarse | run1: PASS 11/11, 2/2; run2: PASS 11/11, 2/2; run3: PASS 11/11, 2/2 |
| Opt 2 fine | run1: PASS 11/11, 2/2; run2: PASS 11/11, 2/2; run3: PASS 11/11, 2/2 |
| Opt 2 coarse | run1: PASS 11/11, 2/2; run2: PASS 11/11, 2/2; run3: PASS 11/11, 2/2 |

### Option 2 vs Option 1 (negative = Option 2 is cheaper)

| Granularity | Opt 1 mean | Opt 2 mean | Difference | Difference % | Difference, successful runs only |
|---|---:|---:|---:|---:|---:|
| fine | $0.0801 | $0.0539 | -$0.0262 | -32.7% | -$0.0262 (-32.7%) |
| coarse | $0.0598 | $0.0359 | -$0.0240 | -40.0% | -$0.0240 (-40.0%) |

**Option 2 is cheaper at coarse granularity** ($0.0359 vs $0.0539) once Jev overhead and escalations are included.

---

## How to interpret your real results

### When routing overhead and escalations cancel the savings

Think in units where an Opus attempt on a step costs **1.0**. From the list prices, a Sonnet attempt costs about **0.5** (2/4 in both input and output) and a Haiku attempt about **0.25**. This assumes equal token counts, which is the simplification to check against your ledger. Opus 5.5 is only 2x Sonnet and 4x Haiku, so even a perfect router caps out at 50% (everything on Sonnet) or 75% (everything on Haiku) savings.

With `q` = the chance an attempt fails its checks:

| Routing | Expected cost per step | Cheaper than Opus-only unless |
|---|---|---|
| Sonnet first, escalate to Opus | 0.5 + q_sonnet x 1.0 | q_sonnet > 0.5 |
| Haiku first, then Sonnet, then Opus | 0.25 + q_haiku x (0.5 + q_sonnet x 1.0) | q_haiku x (0.5 + q_sonnet) > 0.75 |

For example, q_haiku = 0.3 and q_sonnet = 0.1 gives 0.25 + 0.3 x 0.6 = **0.43**, about 57% cheaper. These are model numbers, not measurements. They say the *direct* escalation arithmetic is forgiving, so when savings vanish in practice, look for these instead, in your own report:

1. **Most steps routed to Opus anyway.** Savings can't exceed (share of cost in steps routed down) x (price ratio saved). If Jev sends the expensive steps (core logic, fixes) to Opus and only the cheap ones (scaffold, docs) down, the total barely moves. Read the routing-distribution table weighted by the cost-per-step table, not by step count.
2. **Quality misses that become extra steps.** A cheap model's code that passes the validators but fails tests triggers more `fix_errors` and `select_tests` rounds. Each is a full step with a large prompt. Compare call counts per step type between options. If Option 2 has more fix rounds, that spend is escalation by another name.
3. **More output tokens.** If Sonnet or Haiku emit materially more output tokens for the same step (verbosity, thinking), the 2x or 4x price advantage shrinks, and output is the expensive side. Compare output tokens per step.
4. **Lost cache hits.** Only material if your prefix is large enough to cache (Step 13).
5. **Jev's direct cost, which is probably negligible.** At $0.042 per million input tokens, a ~500-token routing call costs about $0.00002, a fraction of a percent of a step's cost. Check the "Jev share of total cost" column. If it is not tiny, something is wrong (a huge `state`, or a price mismatch). Jev's *latency* is the overhead that actually matters: it adds a serial round trip to every step.
6. **Failed runs.** If Option 2 fails the acceptance tests more often, its cost advantage is not a saving. Use the "successful runs only" column and the success table.

**Rule of thumb:** Option 2 is a real win only if (savings vs Option 1) is bigger than the min-to-max spread of both configurations, **and** it passes the acceptance tests as reliably as Option 1.

### Fine vs. coarse: how to decide from the numbers

Compare the **Option 2 total cost** lines and the "which granularity is cheaper" sentence, then explain the gap with these tables:

- **Fine wins when** the routing-distribution table shows Jev sending real *work* (core, cli, unit_tests, not just scaffold) down a tier **and** that work is accepted without escalation, and the extra routing calls and re-sent context (input tokens per call, number of Jev calls) cost less than the downgrades save. Finer steps give Jev more chances to find cheap work, and each failure is smaller (a failed `models` step wastes less than a failed `scaffold_models + core_edge_cases` group).
- **Coarse wins when** fine's per-step table shows the savings concentrated in a few steps while every step still pays for Jev's round trip and a re-sent prefix, or when merged groups are routed to Sonnet or Opus as a whole and fine's extra boundaries just add consistency failures between sub-steps (look at the "inconsistent with earlier code" escalations).
- **Tie-break on latency and variance.** Fine makes about 3 more Jev calls per run (7 vs 4 code-generation steps) and each is a serial round trip. If costs are within the noise, take the lower latency and the lower success variance.
- **Trust it only with enough runs.** With 3 repeats, a fine-vs-coarse gap smaller than the spread is not a finding. Rerun with `--runs 10` on just the two Option 2 configs: `--configs 2fine,2coarse`.
