# Opus-only vs. Opus + Jev: a cost-comparison harness

This harness runs one Python coding task through two agent setups and reports what each one cost:

- **Option 1 (single model).** Claude Opus 5.5 does everything: planning, writing code, and every small decision.
- **Option 2 (two-model team).** Opus 5.5 plans, writes, and fixes code. Four small, bounded decisions go to
  **Jev**, TypeSafe AI's decision model, as typed Noul/Choice/Score questions. Jev only picks from options the harness
  prepared. If Jev's confidence is below a per-step threshold, the harness asks Opus instead.

Both options share the same agent loop, prompts, tools, and quality gate. The only differences are who answers
the small decisions and, optionally, which model runs the steps routed as simple.

> **Status.** I tested the harness end to end with both APIs stubbed (`python dry_run.py`), and the reference
> solution passes all 11 held-out acceptance tests. It has **not** been run against the live APIs, because this
> environment has no API keys. Any Jev details I could not confirm are marked **UNVERIFIED** in this guide.

### Prices used (looked up 2026-10-05)

| Model | Input $/MTok | Cache write (5m) | Cache read | Output $/MTok | Source |
|---|---|---|---|---|---|
| Claude Opus 5.5 (`claude-opus-5-5`) | 4.00 | 5.00 | 0.20 | 20.00 | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Claude Sonnet 5.5 (optional cheap executor) | 2.00 | 2.50 | 0.20 | 10.00 | same page |
| Jev (`jev-latest`, TypeSafe API) | 0.042 | n/a | n/a | free | The published rate as listed on [OpenRouter](https://openrouter.ai/docs/guides/community/jev). I couldn't confirm TypeSafe's direct price. **Check it in your TypeSafe console.** |

For Jev, the harness uses a `usage.cost` value if the response includes one. Otherwise it prices the call from
the table above using the `usage` token counts, and the report says so.

---

## Step 1: Project setup

**What you need:**

- Python 3.11 or newer.
- The Anthropic SDK (`anthropic` 1.x).
- `httpx`, for Jev.
- `pytest`, which both the agent's code and the quality gate use.

**Jev access.** Jev is called directly on **TypeSafe's API**, `POST https://api.typesafe.ai/v1/systemone`,
with `Authorization: Bearer $TYPESAFE_API_KEY`. Keys come from `console.typesafe.ai/keys`.

**No Jev SDK is used.** I could not verify the method names of TypeSafe's Python SDK, so the harness calls the
REST endpoint directly with `httpx`, in one 5-line method (`Jev._post`). If you'd rather use the SDK, change
only that method. Docs to check:

- TypeSafe API docs: https://docs.typesafe.ai. This environment couldn't reach it, so the request and response
  fields here come from published examples.
- Several look-alike Jev sites exist. Trust TypeSafe's own docs over them.

```text
jev-experiment/
├── run.py                  # entry point: runs both options, writes the report
├── dry_run.py              # same harness with both APIs stubbed (no keys, no spend)
├── requirements.txt
├── .env.example
├── harness/
│   ├── config.py           # model IDs, prices (+ date), thresholds, effort levels
│   ├── ledger.py           # one CallRecord per API call -> calls.jsonl
│   ├── llm.py              # metered Opus + Jev clients
│   ├── decisions.py        # Noul/Choice/Score, OpusDecider, JevDecider (confidence gate)
│   ├── workspace.py        # sandboxed project dir, file tools, pytest runner
│   ├── agent.py            # the shared agent loop
│   ├── quality.py          # own tests + held-out acceptance tests
│   └── report.py           # markdown comparison tables
└── task/
    ├── TASK.md             # the sample coding task (swap in your own)
    ├── starter/            # initial workspace contents (empty)
    └── acceptance/         # held-out tests; the agent never sees these
```

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in both keys, then:
set -a && source .env && set +a
```

`requirements.txt`

```text
anthropic>=1.11,<2
httpx>=0.27
pytest>=8
```

`.env.example`

```bash
# Anthropic API key (Claude Console -> API keys). `ant auth login` also works instead.
ANTHROPIC_API_KEY=sk-ant-...
# Jev, called directly on TypeSafe's API (https://api.typesafe.ai). Keys: console.typesafe.ai/keys
TYPESAFE_API_KEY=...
```

> **Safety:** the agent writes Python code that the harness then runs, through pytest and the CLI. Run the
> harness in a container or a throwaway VM, not on a machine holding credentials you care about.

---

## Step 2: Configuration (models, prices, thresholds)

Everything you would tune is in one file:

- **Prices** each carry a source and the date they were checked.
- **Effort is set explicitly for every step.** Opus 5.5 defaults to `medium`, and leaving it implicit is a common
  source of surprise costs.
- **Jev confidence thresholds are set per step.** Each one reflects what a wrong answer at that step costs to
  recover from.

`harness/config.py`

```python
"""Model IDs, prices, and decision thresholds. Everything you would tune lives here."""

from dataclasses import dataclass

OPUS = "claude-opus-5-5"

# Jev, called directly on TypeSafe's API with TYPESAFE_API_KEY. Once you have tuned the
# confidence thresholds below, replace "jev-latest" with the pinned version ID from
# TypeSafe's docs, so a model upgrade can't silently shift them.
JEV = "jev-latest"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MAX_STATE_CHARS = 100_000  # Jev's request budget is ~32K tokens (~150K chars); stay well under
JEV_MAX_QUESTIONS_PER_CALL = 16  # UNVERIFIED: check TypeSafe's API docs for the real per-request limit

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
    # Jev: $0.042/MTok input, output free -- the published rate as listed on OpenRouter.
    # UNVERIFIED for TypeSafe's direct API: confirm against your TypeSafe console/invoice.
    # The ledger uses `usage.cost` if a response includes it, else this rate.
    JEV: Price(0.042, 0.0, 0.0, 0.0, "OpenRouter listing; confirm in TypeSafe console", "2026-10-05"),
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
```

---

## Step 3: The cost ledger

Every API call becomes one `CallRecord`, holding:

- option, run, and step type
- the model that actually served the call
- uncached input, cache-write, cache-read, and output tokens
- latency and cost
- for Jev calls, how many questions were asked and how many came back below threshold

The records are written to `runs/<timestamp>/calls.jsonl`, so you can re-analyze them later without
re-running anything.

`harness/ledger.py`

```python
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
```

---

## Step 4: Metered clients for Opus and Jev

**`Opus.create` wraps `client.beta.messages.create`.** It reads cost straight from `response.usage`:

- `input_tokens` (uncached)
- `cache_creation_input_tokens`
- `cache_read_input_tokens`
- `output_tokens` (thinking tokens are billed as output)

**Settings used on every Opus call:**

- adaptive thinking
- an explicit effort level
- automatic prompt caching, identical in both options
- server-side refusal fallback (`fallbacks: "default"`)

If a fallback ever serves a turn, the ledger prices it by `response.model` and tags the record.

**`Jev.decide` posts `{model, state, questions}`** to TypeSafe's `/v1/systemone` endpoint. It records
`usage.input_tokens` and `usage.output_tokens`, plus `usage.cost` if the response includes one.

**UNVERIFIED:**

- The `answers` wrapper around the per-question results.
- The `usage` field names.
- Whether the response includes a cost.

These come from published examples of the API, not from TypeSafe's docs page, which this environment couldn't
reach. If the response shape differs, the harness raises an error that lists the keys it actually got.

`harness/llm.py`

```python
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
                "Unexpected Jev API response shape (no 'answers' key; got "
                f"{sorted(body)}). Check the current response schema in the docs."
            )
        return body["answers"], rec
```

---

## Step 5: The decision layer (where options are prepared and confidence decides)

This is the core of Option 2. Each decision point builds typed questions, and the options are written by your
code, not by a model:

- **Noul:** yes/no, with a description of what `true` and `false` mean.
- **Choice:** one label from a dict of `label -> description` that the harness wrote.
- **Score:** a position on an ordered list of 2–10 levels.

**How the two deciders work:**

- `OpusDecider` (Option 1, and Option 2's fallback) turns the same questions into a prompt plus a JSON schema.
  Structured outputs guarantee the answer is one of the prepared options.
- `JevDecider` (Option 2) sends the questions to Jev. **The confidence gate** keeps answers at or above
  `JEV_MIN_CONFIDENCE[step]` and re-asks Opus only for the rest, in one call. If Jev errors, the whole batch
  falls back to Opus.

Noul answers come with no `confidence` field, only a probability `p`. The harness uses `|2p − 1|` as the
confidence: 0 at `p = 0.5` ("can't tell") and 1 at certainty.

**UNVERIFIED in this file:**

- Request field names: the harness sends `criteria` as a dict for Choice and as a list for Score, as in the
  examples I found.
- The maximum number of questions per request (`JEV_MAX_QUESTIONS_PER_CALL = 16` is a placeholder).

`harness/decisions.py`

```python
"""Small, bounded decisions. The harness prepares every option; a decider only picks.

Option 1 uses OpusDecider for everything. Option 2 uses JevDecider, which asks Jev first
and re-asks Opus only for the answers whose confidence is below the step's threshold.
"""

import json
from dataclasses import dataclass

from .config import EFFORT, JEV_MAX_QUESTIONS_PER_CALL, JEV_MAX_STATE_CHARS, JEV_MIN_CONFIDENCE
from .llm import Jev, Opus


# --- Question types (mirror Jev's Noul / Choice / Score) ---------------------------------

@dataclass
class Noul:
    """Yes/no."""
    instructions: str
    true: str
    false: str


@dataclass
class Choice:
    """Pick exactly one label from options the harness prepared (label -> description)."""
    instructions: str
    options: dict[str, str]


@dataclass
class Score:
    """Place the state on an ordered scale of 2-10 levels; levels[0] is the lowest."""
    instructions: str
    levels: list[str]


Question = Noul | Choice | Score


@dataclass
class Answer:
    value: bool | str | float  # Noul -> bool, Choice -> label, Score -> level position (0..n-1)
    confidence: float | None  # None for Opus: it gives no calibrated confidence
    by: str  # "jev" | "opus"


# --- Jev wire format -------------------------------------------------------------------------
# Request/response field names below follow published examples of TypeSafe's /v1/systemone API
# as of 2026-10-05 (docs.typesafe.ai was not reachable to confirm). Re-check before relying on this.

def to_jev(q: Question) -> dict:
    if isinstance(q, Noul):
        return {"type": "noul", "instructions": q.instructions,
                "criteria": {"true": q.true, "false": q.false}}
    if isinstance(q, Choice):
        return {"type": "choice", "instructions": q.instructions, "criteria": q.options}
    return {"type": "score", "instructions": q.instructions, "criteria": q.levels}


def from_jev(q: Question, a: dict) -> Answer:
    if isinstance(q, Noul):
        p = float(a["noul"])  # P(true); 0.5 means "can't tell"
        return Answer(p >= 0.5, abs(2 * p - 1), "jev")
    if isinstance(q, Choice):
        return Answer(a["choice"], float(a["confidence"]), "jev")
    return Answer(float(a["score"]), float(a["confidence"]), "jev")


# --- Deciders --------------------------------------------------------------------------------

DECIDER_SYSTEM = (
    "You answer small, bounded questions for a coding agent's harness. "
    "Use only the options given. Answer every question."
)


class OpusDecider:
    def __init__(self, opus: Opus):
        self.opus = opus

    def decide(self, step: str, state: str, questions: dict[str, Question], note: str = "") -> dict[str, Answer]:
        props, lines = {}, []
        for key, q in questions.items():
            if isinstance(q, Noul):
                props[key] = {"type": "boolean"}
                lines.append(f"- {key} (true/false): {q.instructions}\n  true = {q.true}\n  false = {q.false}")
            elif isinstance(q, Choice):
                props[key] = {"type": "string", "enum": list(q.options)}
                opts = "\n".join(f"  {label}: {desc}" for label, desc in q.options.items())
                lines.append(f"- {key} (pick one label): {q.instructions}\n{opts}")
            else:
                props[key] = {"type": "integer", "enum": list(range(len(q.levels)))}
                lvls = "\n".join(f"  {i}: {desc}" for i, desc in enumerate(q.levels))
                lines.append(f"- {key} (pick a level): {q.instructions}\n{lvls}")
        schema = {"type": "object", "properties": props, "required": list(props),
                  "additionalProperties": False}
        prompt = f"<state>\n{state}\n</state>\n\nQuestions:\n" + "\n".join(lines)

        resp = self.opus.create(step, system=DECIDER_SYSTEM, effort=EFFORT["decide"],
                                messages=[{"role": "user", "content": prompt}],
                                json_schema=schema, max_tokens=4000, note=note)
        text = next(b.text for b in resp.content if b.type == "text")
        raw = json.loads(text)
        return {k: Answer(raw[k], None, "opus") for k in questions}


class JevDecider:
    def __init__(self, jev: Jev, fallback: OpusDecider):
        self.jev, self.fallback = jev, fallback

    def decide(self, step: str, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        state = state[:JEV_MAX_STATE_CHARS]
        threshold = JEV_MIN_CONFIDENCE[step]
        answers: dict[str, Answer] = {}
        unsure: dict[str, Question] = {}

        keys = list(questions)
        for i in range(0, len(keys), JEV_MAX_QUESTIONS_PER_CALL):
            batch = {k: questions[k] for k in keys[i:i + JEV_MAX_QUESTIONS_PER_CALL]}
            try:
                raw, rec = self.jev.decide(step, state, {k: to_jev(q) for k, q in batch.items()})
            except Exception as e:  # network error, 4xx/5xx, schema drift: Opus decides the batch
                print(f"  [jev] {step}: {type(e).__name__}: {e} -> falling back to Opus")
                unsure.update(batch)
                continue
            for k, q in batch.items():
                a = from_jev(q, raw[k])
                if a.confidence >= threshold:
                    answers[k] = a
                else:
                    unsure[k] = q
                    rec.low_confidence += 1

        # The confidence gate: anything Jev was unsure about goes to Opus, in one call.
        if unsure:
            answers.update(self.fallback.decide(step, state, unsure, note="jev-fallback"))
        return answers
```

---

## Step 6: Workspace and tools

Each run gets a fresh project directory. The executor has two strict tools, `read_file` and `write_file`, and
paths can't escape the workspace. Tools are listed in sorted order so the cached prefix stays byte-identical.
`pytest` exit codes feed the recovery step.

`harness/workspace.py`

```python
"""The agent's sandboxed project directory: file tools and pytest runs."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import PYTEST_TIMEOUT_S

IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".venv", ".git"}

TOOLS = [  # sorted by name so the tools prefix is byte-stable for prompt caching
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the project directory.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to the project root"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a UTF-8 text file in the project directory.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the project root"},
                "content": {"type": "string", "description": "Full file contents"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class TestRun:
    ok: bool
    exit_code: int  # pytest: 0 ok, 1 failures, 2 interrupted, 3 internal, 4 usage, 5 none collected
    output: str
    timed_out: bool = False


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError(f"path escapes the workspace: {rel}")
        return p

    def files(self) -> list[str]:
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file() and not IGNORED_DIRS & set(p.relative_to(self.root).parts)
        )

    def test_files(self) -> list[str]:
        return [f for f in self.files() if f.startswith("tests/") and f.rsplit("/", 1)[-1].startswith("test_")]

    def read(self, rel: str) -> str:
        return self._path(rel).read_text()

    def write(self, rel: str, content: str) -> None:
        p = self._path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def head(self, rel: str, n: int = 30) -> str:
        return "\n".join(self.read(rel).splitlines()[:n])

    def pytest(self, targets: list[str] | None = None) -> TestRun:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *(targets or [])]
        try:
            r = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=PYTEST_TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            return TestRun(False, -1, out + "\n[timed out]", timed_out=True)
        return TestRun(r.returncode == 0, r.returncode, r.stdout + r.stderr)
```

---

## Step 7: The agent loop (shared by both options)

Opus does the hard reasoning: the plan, the code, and the fixes. The four small decisions go through
`self.decider`:

| Step | Question type | Options prepared by the harness | Jev threshold |
|---|---|---|---|
| `context`: which files are relevant? | Noul per file | the current file list, plus the first 30 lines of each file | 0.6 |
| `route`: is this step simple enough for a cheaper setting? | Score (3 levels) | fixed rubric: Mechanical / Moderate / Hard | 0.7 |
| `tests`: which focused tests run first? | Noul per test file | the current `tests/test_*.py` files | 0.6 |
| `recovery`: which recovery path after a failure? | Choice | 2 labels chosen by `recovery_options()` from the pytest exit code | 0.8 |

**What "simple" buys you.** In both options a simple step runs on Opus at `low` effort. In Option 2 you can also
pass `--cheap-model claude-sonnet-5-5`, which sends simple steps to Sonnet 5.5 instead. That flag is off by
default so the baseline comparison isolates Jev's effect. Run it as a separate experiment.

**Test selection still ends with the full suite.** Focused tests run first, then the full suite. A wrong test
pick can delay finding a failure, but it can't hide one.

`harness/agent.py`

```python
"""The agent loop shared by both options. Only the `decider` (and optional cheap model) differ.

    plan (Opus)
    for each subtask:
        context   <- decider: which files are relevant?            (Noul per file)
        route     <- decider: simple or complex?                   (Score)
        execute   (Opus, or the cheap model for simple steps in Option 2)
        loop:
            tests    <- decider: which focused tests run first?    (Noul per test file)
            run focused tests, then the full suite
            recovery <- decider: which recovery path?              (Choice from prepared options)
            fix      (Opus)
"""

import json

from .config import EFFORT, MAX_FIX_ATTEMPTS, MAX_TOOL_TURNS, OPUS
from .decisions import Choice, Noul, Score
from .llm import Opus
from .workspace import TOOLS, TestRun, Workspace

SYSTEM = """You are a senior Python engineer working in an empty-to-start project directory.
Use Python 3.11 and the standard library only; tests use pytest and live in tests/.
Read and write files only with the read_file and write_file tools; paths are relative to the
project root. Do only the subtask you are given. When it is done, reply with a one-line
summary and no tool calls."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "detail": {"type": "string"}},
                "required": ["title", "detail"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subtasks"],
    "additionalProperties": False,
}

# Recovery paths the harness offers, keyed by label. The decider only picks among these.
RECOVERY = {
    "fix_code": "The implementation is wrong or incomplete; edit source files.",
    "fix_tests": "The tests are wrong or contradict the spec; edit test files.",
    "fix_environment": "Import, collection, or project-layout problem (package path, __init__.py, conftest).",
    "rerun": "Transient problem such as a timeout unrelated to the code; rerun unchanged.",
}


def recovery_options(run: TestRun) -> dict[str, str]:
    """Prepare the candidate recovery paths for this kind of failure."""
    if run.timed_out:
        keys = ["rerun", "fix_code"]
    elif run.exit_code in (2, 3, 4):  # collection / internal / usage error
        keys = ["fix_environment", "fix_code"]
    elif run.exit_code == 5:  # no tests collected: nothing to decide
        keys = ["fix_tests"]
    else:
        keys = ["fix_code", "fix_tests"]
    return {k: RECOVERY[k] for k in keys}


class Agent:
    def __init__(self, opus: Opus, decider, ws: Workspace, spec: str, cheap_model: str | None = None):
        self.opus, self.decider, self.ws, self.spec, self.cheap_model = opus, decider, ws, spec, cheap_model

    def run(self) -> bool:
        for st in self.plan():
            print(f"  subtask: {st['title']}")
            context = self.choose_context(st)
            model, effort = self.route(st)
            changed = self.execute("execute", st, context, model, effort)
            self.verify(st, context | changed)
        return self.ws.pytest().ok

    # --- Opus: planning and code ---------------------------------------------------------------

    def plan(self) -> list[dict]:
        resp = self.opus.create(
            "plan", system=SYSTEM, effort=EFFORT["plan"], json_schema=PLAN_SCHEMA,
            messages=[{"role": "user", "content": f"{self.spec}\n\nBreak this task into 3-6 ordered "
                       "subtasks. Each subtask must leave the project in a state where its tests pass."}],
        )
        return json.loads(next(b.text for b in resp.content if b.type == "text"))["subtasks"]

    def execute(self, step: str, st: dict, files: set[str], model: str, effort: str, extra: str = "") -> set[str]:
        """Run one tool-use loop for a subtask. Returns the paths written."""
        listing = "\n".join(self.ws.files()) or "(empty)"
        context = "\n\n".join(f"<file path='{f}'>\n{self.ws.read(f)}\n</file>"
                              for f in sorted(files) if f in self.ws.files())
        user = (f"<spec>\n{self.spec}\n</spec>\n\n<project_files>\n{listing}\n</project_files>\n\n"
                f"{context}\n\nSubtask: {st['title']}\n{st['detail']}\n{extra}")
        messages = [{"role": "user", "content": user}]
        written: set[str] = set()

        for _ in range(MAX_TOOL_TURNS):
            resp = self.opus.create(step, model=model, effort=effort, system=SYSTEM,
                                    messages=messages, tools=TOOLS)
            messages.append({"role": "assistant", "content": resp.content})  # append-only history
            if resp.stop_reason != "tool_use":
                break
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                try:
                    if block.name == "write_file":
                        self.ws.write(block.input["path"], block.input["content"])
                        written.add(block.input["path"])
                        out = "ok"
                    else:
                        out = self.ws.read(block.input["path"])
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
                except Exception as e:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": f"{type(e).__name__}: {e}", "is_error": True})
            messages.append({"role": "user", "content": results})
        return written

    # --- Small decisions: delegated to the decider -------------------------------------------

    def choose_context(self, st: dict) -> set[str]:
        files = self.ws.files()
        if not files:
            return set()
        heads = "\n\n".join(f"### {f}\n{self.ws.head(f)}" for f in files)
        state = f"Subtask: {st['title']}\n{st['detail']}\n\nProject files (first lines of each):\n{heads}"
        questions = {
            f"f{i}": Noul(f"Should `{f}` be given to the engineer as context for this subtask?",
                          true="The subtask edits this file or calls code defined in it",
                          false="The subtask neither edits nor depends on this file")
            for i, f in enumerate(files)
        }
        answers = self.decider.decide("context", state, questions)
        return {f for i, f in enumerate(files) if answers[f"f{i}"].value}

    def route(self, st: dict) -> tuple[str, str]:
        state = f"Subtask: {st['title']}\n{st['detail']}"
        q = {"difficulty": Score("How much reasoning does this coding subtask need?", [
            "Mechanical: boilerplate, packaging, a trivial function, or wiring existing code",
            "Moderate: a small function with a few edge cases",
            "Hard: non-trivial logic, design decisions, or subtle edge cases",
        ])}
        level = float(self.decider.decide("route", state, q)["difficulty"].value)
        if level < 0.5:  # "Mechanical"
            return (self.cheap_model or OPUS), EFFORT["execute_simple"]
        return OPUS, EFFORT["execute_complex"]

    def select_tests(self, st: dict, changed: set[str]) -> list[str]:
        tests = self.ws.test_files()
        if len(tests) <= 1:
            return tests  # nothing to choose between
        heads = "\n\n".join(f"### {t}\n{self.ws.head(t)}" for t in tests)
        state = (f"Subtask: {st['title']}\nChanged files: {', '.join(sorted(changed)) or 'none'}\n\n"
                 f"Test files (first lines of each):\n{heads}")
        questions = {
            f"t{i}": Noul(f"Should `{t}` run first to check this subtask's changes?",
                          true="It directly exercises code changed in this subtask",
                          false="It does not exercise the changed code")
            for i, t in enumerate(tests)
        }
        answers = self.decider.decide("tests", state, questions)
        return [t for i, t in enumerate(tests) if answers[f"t{i}"].value]

    def recover(self, st: dict, run: TestRun) -> str:
        options = recovery_options(run)
        if len(options) == 1:
            return next(iter(options))
        state = (f"Subtask: {st['title']}\n{st['detail']}\n\npytest exit code {run.exit_code}. "
                 f"Output (tail):\n{run.output[-4000:]}")
        q = {"action": Choice("Which recovery path fits this test failure?", options)}
        return self.decider.decide("recovery", state, q)["action"].value

    # --- Verify / recover loop -----------------------------------------------------------------

    def verify(self, st: dict, changed: set[str]) -> bool:
        focused, seen_tests = [], None
        for attempt in range(MAX_FIX_ATTEMPTS + 1):
            if self.ws.test_files() != seen_tests:  # re-decide only when the test set changes
                seen_tests = self.ws.test_files()
                focused = self.select_tests(st, changed)
            run = self.ws.pytest(focused)
            if run.ok and focused:
                run = self.ws.pytest()  # focused tests pass: widen to the full suite
            if run.ok:
                return True
            if attempt == MAX_FIX_ATTEMPTS:
                print(f"  gave up on: {st['title']}")
                return False
            action = self.recover(st, run)
            print(f"  recovery: {action}")
            if action == "rerun":
                continue
            extra = (f"\nThe tests are failing. Recovery path chosen: {action} - {RECOVERY[action]}\n"
                     f"pytest output (tail):\n{run.output[-4000:]}")
            changed |= self.execute("fix", st, changed, OPUS, EFFORT["execute_complex"], extra)
        return False
```

---

## Step 8: Wire up Option 1 and Option 2, then run

`run_once` is the only place the two options differ. `--repeats` runs each option several times, and the order
alternates each repeat, so neither option always runs first: against warm caches, or into the same rate-limit
window.

`run.py`

```python
"""Run the same coding task through Option 1 (Opus only) and Option 2 (Opus + Jev), then report.

    python run.py                       # both options, one run each
    python run.py --repeats 3           # 3 runs per option (recommended minimum)
    python run.py --options 2 --cheap-model claude-sonnet-5-5
"""

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

from harness import quality
from harness.agent import Agent
from harness.decisions import JevDecider, OpusDecider
from harness.ledger import Ledger
from harness.llm import Jev, Opus
from harness.report import RunResult, render
from harness.workspace import Workspace

ROOT = Path(__file__).resolve().parent


def run_once(option: str, run: int, spec: str, out_dir: Path, ledger: Ledger, cheap_model: str | None) -> RunResult:
    ws_root = out_dir / f"{option}-run{run}" / "workspace"
    shutil.copytree(ROOT / "task" / "starter", ws_root)
    ws = Workspace(ws_root)

    opus = Opus(ledger, option, run)
    if option == "option1":
        decider, cheap = OpusDecider(opus), None  # Opus makes every decision and writes every step
    else:
        decider, cheap = JevDecider(Jev(ledger, option, run), OpusDecider(opus)), cheap_model

    print(f"== {option} run {run} ==")
    t0 = time.perf_counter()
    error = ""
    try:
        Agent(opus, decider, ws, spec, cheap_model=cheap).run()
    except Exception as e:  # keep the run in the report; quality check below decides success
        error = f"{type(e).__name__}: {e}"
        print(f"  error: {error}")
    wall = time.perf_counter() - t0
    q = quality.check(ws_root)
    print(f"  acceptance {q.acceptance_passed}/{q.acceptance_total}, own tests {q.own_passed}/{q.own_total}")
    return RunResult(option, run, q, wall, error)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--options", nargs="+", choices=["1", "2"], default=["1", "2"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--task", type=Path, default=ROOT / "task" / "TASK.md")
    ap.add_argument("--cheap-model", default=None,
                    help="Option 2 only: model for steps Jev routes as simple (e.g. claude-sonnet-5-5)")
    args = ap.parse_args()

    spec = args.task.read_text()
    out_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    ledger = Ledger(out_dir / "calls.jsonl")
    results = []
    try:
        for run in range(args.repeats):
            # Alternate order each repeat so neither option always runs first (rate limits, warm caches).
            order = args.options if run % 2 == 0 else list(reversed(args.options))
            for o in order:
                results.append(run_once(f"option{o}", run, spec, out_dir, ledger, args.cheap_model))
    finally:
        ledger.save()

    report = render(ledger.records, results)
    (out_dir / "report.md").write_text(report)
    print("\n" + report)
    print(f"Saved {out_dir / 'report.md'} and {ledger.path}")


if __name__ == "__main__":
    main()
```

---

## Step 9: Quality check (so the comparison is fair)

A cheaper run only counts if the code still works. The harness measures two things after each run:

1. **The agent's own test suite.** This is informative, but the agent wrote these tests, so it is not enough on
   its own.
2. **Held-out acceptance tests** in `task/acceptance/`. They run the CLI as a subprocess against the agent's
   workspace, and the agent never sees them. **A run succeeds only if every acceptance test passes.** Compare
   costs only between successful runs.

`harness/quality.py`

```python
"""Quality gate: the agent's own tests plus held-out acceptance tests it never sees."""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .config import PYTEST_TIMEOUT_S

ACCEPTANCE_DIR = Path(__file__).resolve().parent.parent / "task" / "acceptance"


@dataclass
class Quality:
    own_passed: int
    own_total: int
    acceptance_passed: int
    acceptance_total: int

    @property
    def succeeded(self) -> bool:
        return self.acceptance_total > 0 and self.acceptance_passed == self.acceptance_total


def _pytest_counts(args: list[str], cwd: Path, xml_path: Path, env: dict | None = None) -> tuple[int, int]:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml_path}", *args]
    try:
        subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=PYTEST_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return 0, 0
    if not xml_path.exists():
        return 0, 0
    root = ET.parse(xml_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    total = int(suite.get("tests", 0))
    bad = sum(int(suite.get(k, 0)) for k in ("failures", "errors", "skipped"))
    return total - bad, total


def check(ws_root: Path) -> Quality:
    own = _pytest_counts([], ws_root, ws_root.parent / "own-tests.xml")
    env = {**os.environ, "HARNESS_WORKSPACE": str(ws_root)}
    acc = _pytest_counts([str(ACCEPTANCE_DIR)], ACCEPTANCE_DIR, ws_root.parent / "acceptance.xml", env)
    return Quality(*own, *acc)
```

**The sample task** is your example: a CLI that flags duplicate orders in a CSV, with unit tests. The spec pins
down the exact contract, so the held-out tests can check it. To use your own task, swap `TASK.md` and the
acceptance tests.

`task/TASK.md`

````markdown
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
````

There are 11 held-out tests in total. Here is an excerpt (full file: `task/acceptance/test_acceptance.py`):

```python
def test_duplicate_by_content_normalized(tmp_path):
    r = run_cli(tmp_path, HEADER
                + "1,Alice@Example.com,sku-7,2,2026-03-01,web\n"
                + "2, alice@example.COM ,SKU-7,02,2026-03-01,app\n")
    assert r.returncode == 0, r.stderr
    out = rows(r.stdout)
    assert out[1]["is_duplicate"] == "true"
    assert out[1]["duplicate_of"] == "1"


def test_missing_required_column(tmp_path):
    r = run_cli(tmp_path, "order_id,customer_email,sku,order_date\n1,a@x.com,SKU1,2026-01-01\n")
    assert r.returncode == 2
    assert "missing column" in r.stderr.lower()
    assert "quantity" in r.stderr
```

---

## Step 10: The comparison report

`report.py` produces four tables:

1. **Summary:** success, cost, call counts, tokens, and latency, as a mean per run with a min–max range.
2. **Cost per step type.**
3. **Per option × step × model detail.**
4. **The prices used, with source and date.**

It also says what share of Jev's answers passed the confidence gate.

`harness/report.py`

```python
"""Turn the call ledger and quality results into a markdown comparison report."""

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from .config import JEV, PRICES
from .ledger import CallRecord
from .quality import Quality

STEPS = ["plan", "context", "route", "execute", "tests", "recovery", "fix"]
LABEL = {"option1": "Option 1 (Opus only)", "option2": "Option 2 (Opus + Jev)"}


@dataclass
class RunResult:
    option: str
    run: int
    quality: Quality | None
    wall_s: float
    error: str = ""


def _fmt_range(vals: list[float], fmt: str) -> str:
    m, lo, hi = (format(v, fmt) for v in (mean(vals), min(vals), max(vals)))
    return m if lo == hi else f"{m} ({lo}–{hi})"


def _delta(a: float, b: float) -> str:
    return "—" if a == 0 else f"{(b - a) / a:+.0%}"


def render(records: list[CallRecord], results: list[RunResult]) -> str:
    options = sorted({r.option for r in results})
    runs = {o: sorted({r.run for r in results if r.option == o}) for o in options}

    def per_run(o: str, f) -> list[float]:
        return [sum(f(c) for c in records if c.option == o and c.run == r) for r in runs[o]]

    is_opus = lambda c: c.model != JEV  # noqa: E731
    rows: list[tuple[str, dict[str, list[float]], str]] = [
        ("Total cost per run (USD)", {o: per_run(o, lambda c: c.cost_usd) for o in options}, ".4f"),
        ("Claude calls per run", {o: per_run(o, lambda c: is_opus(c)) for o in options}, ".1f"),
        ("Jev calls per run", {o: per_run(o, lambda c: not is_opus(c)) for o in options}, ".1f"),
        ("Opus fallback calls (Jev unsure or unavailable)", {o: per_run(o, lambda c: c.note == "jev-fallback") for o in options}, ".1f"),
        ("Claude input tokens (uncached)", {o: per_run(o, lambda c: c.input_tokens * is_opus(c)) for o in options}, ",.0f"),
        ("Claude cache-read tokens", {o: per_run(o, lambda c: c.cache_read_tokens) for o in options}, ",.0f"),
        ("Claude cache-write tokens", {o: per_run(o, lambda c: c.cache_write_tokens) for o in options}, ",.0f"),
        ("Claude output tokens (incl. thinking)", {o: per_run(o, lambda c: c.output_tokens * is_opus(c)) for o in options}, ",.0f"),
        ("Jev input tokens", {o: per_run(o, lambda c: c.input_tokens * (not is_opus(c))) for o in options}, ",.0f"),
        ("Model latency per run (s, summed)", {o: per_run(o, lambda c: c.latency_s) for o in options}, ".1f"),
        ("Wall-clock per run (s)", {o: [r.wall_s for r in results if r.option == o] for o in options}, ".1f"),
    ]

    out = ["# Cost comparison\n"]
    n_runs = {o: len(runs[o]) for o in options}
    out.append(f"Runs per option: {', '.join(f'{LABEL[o]}: {n_runs[o]}' for o in options)}. "
               "Values are means per run; ranges in parentheses when runs > 1.\n")

    header = "| Metric | " + " | ".join(LABEL[o] for o in options) + (" | Δ (2 vs 1) |" if len(options) == 2 else " |")
    out += [header, "|" + "---|" * (header.count("|") - 1)]

    def quality_cell(o: str) -> str:
        rs = [r for r in results if r.option == o]
        ok = sum(1 for r in rs if r.quality and r.quality.succeeded)
        acc = [f"{r.quality.acceptance_passed}/{r.quality.acceptance_total}" if r.quality else "error" for r in rs]
        return f"{'✅' if ok == len(rs) else '❌'} {ok}/{len(rs)} runs (acceptance {', '.join(acc)})"

    def own_cell(o: str) -> str:
        return ", ".join(f"{r.quality.own_passed}/{r.quality.own_total}" if r.quality else "error"
                         for r in results if r.option == o)

    extra = " | |" if len(options) == 2 else " |"
    out.append("| Task succeeded (held-out acceptance tests) | " + " | ".join(quality_cell(o) for o in options) + extra)
    out.append("| Agent's own tests passing | " + " | ".join(own_cell(o) for o in options) + extra)
    for name, vals, fmt in rows:
        cells = [_fmt_range(vals[o], fmt) for o in options]
        d = f" | {_delta(mean(vals[options[0]]), mean(vals[options[1]]))} |" if len(options) == 2 else " |"
        out.append(f"| {name} | " + " | ".join(cells) + d)

    jev_calls = [c for c in records if c.model == JEV]
    if jev_calls:
        asked = sum(c.questions for c in jev_calls)
        low = sum(c.low_confidence for c in jev_calls)
        out.append(f"\nJev answered {asked} questions; {asked - low} accepted ({(asked - low) / asked:.0%}), "
                   f"{low} below threshold and re-asked to Opus.\n")

    # Cost per step type
    out.append("\n## Cost per step type (mean per run)\n")
    out.append("| Step | " + " | ".join(f"{LABEL[o]} cost | calls (Claude/Jev) | latency s" for o in options) + " |")
    out.append("|---|" + "---|---|---|" * len(options))
    for step in STEPS:
        cells = []
        for o in options:
            cs = [c for c in records if c.option == o and c.step == step]
            n = n_runs[o]
            cells.append(f"${sum(c.cost_usd for c in cs) / n:.4f} | "
                         f"{sum(is_opus(c) for c in cs) / n:.1f}/{sum(not is_opus(c) for c in cs) / n:.1f} | "
                         f"{sum(c.latency_s for c in cs) / n:.1f}")
        out.append(f"| {step} | " + " | ".join(cells) + " |")

    # Per model per step detail (totals across runs)
    out.append("\n## Detail: per option × step × model (totals across runs)\n")
    out.append("| Option | Step | Model | Calls | Input | Cache read | Cache write | Output | Latency s | Cost USD |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    agg: dict[tuple, list[CallRecord]] = defaultdict(list)
    for c in records:
        agg[(c.option, c.step, c.model)].append(c)
    for (o, step, model), cs in sorted(agg.items(), key=lambda kv: (kv[0][0], STEPS.index(kv[0][1]), kv[0][2])):
        out.append(f"| {o} | {step} | {model} | {len(cs)} | {sum(c.input_tokens for c in cs):,} | "
                   f"{sum(c.cache_read_tokens for c in cs):,} | {sum(c.cache_write_tokens for c in cs):,} | "
                   f"{sum(c.output_tokens for c in cs):,} | {sum(c.latency_s for c in cs):.1f} | "
                   f"{sum(c.cost_usd for c in cs):.4f} |")

    # Prices used
    used = sorted({c.model for c in records})
    out.append("\n## Prices used (USD per million tokens)\n")
    out.append("| Model | Input | Cache write (5m) | Cache read | Output | Source | Checked |")
    out.append("|---|---|---|---|---|---|---|")
    for m in used:
        p = PRICES[m]
        out.append(f"| {m} | {p.input} | {p.cache_write_5m} | {p.cache_read} | {p.output} | {p.source} | {p.checked} |")
    if any(c.note == "cost-estimated-from-price-table" for c in jev_calls):
        out.append("\nSome Jev calls had no `usage.cost` in the response; those were priced from the table above.")
    elif jev_calls:
        out.append("\nJev costs are the `usage.cost` values the Jev API reported per call.")

    errors = [r for r in results if r.error]
    if errors:
        out.append("\n## Errors\n")
        out += [f"- {r.option} run {r.run}: {r.error}" for r in errors]
    return "\n".join(out) + "\n"
```

---

## Step 11: Dry run first, then the real run

```bash
python dry_run.py                     # stubs both APIs: checks wiring, report, acceptance tests (no spend)
python run.py --repeats 3             # live: 3 runs per option
python run.py --options 2 --cheap-model claude-sonnet-5-5 --repeats 3   # optional separate experiment
```

Each live run writes `runs/<timestamp>/report.md`, `calls.jsonl`, and every workspace. That lets you inspect
the code each option actually produced.

---

## Example final report

> **Illustrative numbers. These were not measured.** They show the shape of the report and plausible magnitudes
> for this task. Your real numbers will differ. Run the harness to get them.

| Metric | Option 1 (Opus only) | Option 2 (Opus + Jev) | Δ (2 vs 1) |
|---|---|---|---|
| Task succeeded (held-out acceptance tests) | ✅ 3/3 runs (acceptance 11/11, 11/11, 11/11) | ✅ 3/3 runs (acceptance 11/11, 11/11, 11/11) | |
| Agent's own tests passing | 14/14, 12/12, 15/15 | 13/13, 14/14, 12/12 | |
| Total cost per run (USD) | 1.4120 (1.2210–1.6630) | 1.1870 (1.0420–1.4110) | -16% |
| Claude calls per run | 41.3 | 24.1 | -42% |
| Jev calls per run | 0.0 | 19.0 | — |
| Opus fallback calls (Jev unsure or unavailable) | 0.0 | 2.7 | — |
| Claude input tokens (uncached) | 96,400 | 78,900 | -18% |
| Claude cache-read tokens | 412,000 | 351,000 | -15% |
| Claude cache-write tokens | 88,300 | 79,100 | -10% |
| Claude output tokens (incl. thinking) | 38,900 | 33,700 | -13% |
| Jev input tokens | 0 | 31,500 | — |
| Model latency per run (s, summed) | 412.6 | 318.9 | -23% |
| Wall-clock per run (s) | 431.0 (388.2–497.5) | 334.4 (301.7–380.0) | -22% |

Jev answered 142 questions: 126 accepted (89%), and 16 were below threshold and re-asked to Opus.

**Cost per step type** (mean per run):

| Step | Option 1 cost | calls (Claude/Jev) | latency s | Option 2 cost | calls (Claude/Jev) | latency s |
|---|---|---|---|---|---|---|
| plan | $0.0610 | 1.0/0.0 | 38.2 | $0.0590 | 1.0/0.0 | 36.9 |
| context | $0.0720 | 5.0/0.0 | 31.0 | $0.0110 | 0.7/5.0 | 6.1 |
| route | $0.0460 | 5.0/0.0 | 24.5 | $0.0090 | 0.7/5.0 | 4.8 |
| execute | $0.8030 | 15.3/0.0 | 228.4 | $0.7860 | 14.7/0.0 | 221.0 |
| tests | $0.0810 | 7.0/0.0 | 33.1 | $0.0120 | 1.3/7.0 | 7.4 |
| recovery | $0.0290 | 2.0/0.0 | 11.2 | $0.0010 | 0.0/2.0 | 0.6 |
| fix | $0.3200 | 6.0/0.0 | 46.2 | $0.3090 | 5.7/0.0 | 42.1 |

---

## How to interpret the results

1. **Check quality first.** Compare cost only between options that passed the acceptance tests on every run.
   If Option 2 fails more often, its lower cost per run is not a saving, because you'd pay for retries or
   manual fixes. Compare **cost per successful run** instead.
2. **The savings are capped by the decision steps' share of the bill.** Look at the step table. Jev can
   eliminate at most the `context + route + tests + recovery` rows of Option 1, about 16% in the illustrative
   example. `execute` and `fix` dominate, and Jev doesn't touch them. If your task has few decision points per
   unit of code, expect small savings.
3. **Watch for second-order effects in `execute` and `fix`.** A different context choice or recovery path
   changes what Opus does next. If Option 2's `fix` row is clearly higher, Jev's decisions are costing you
   downstream. Tighten that step's threshold, or give Jev better-described options.
4. **Use the fallback rate to tune thresholds.** A high re-ask rate (say, above 30%) means you're paying for
   both models on many decisions. Lower the threshold where a wrong answer is cheap to recover from (`context`,
   `tests`). Raise it where a wrong answer is expensive (`recovery`, `route` once a cheap model is enabled).
5. **Treat small differences as noise.** Agent runs aren't deterministic. If the min–max ranges of the two
   options overlap, the result is inconclusive. Use at least 3 repeats, and preferably 5–10, before drawing
   conclusions.
6. **Latency often improves more than cost.** Each Opus decision call includes thinking. A Jev call is a
   single forward pass. Even when the dollar savings are modest, wall-clock time can drop noticeably.

## Factors that can distort the comparison

- **Prompt caching.**
  - Caches are model-scoped and expire (5-minute TTL by default).
  - Decision calls use a different system prompt from the executor, so they never share its cache.
  - In Option 1, the many small Opus decision calls write and read their own cache entries, which Option 2
    mostly doesn't have.
  - Prefixes below the model's minimum cacheable length don't cache at all.
  - Check the cache-read and cache-write rows: a large gap between the options can explain the cost
    difference by itself.
- **Cache invalidation from model switching.** With `--cheap-model`, simple steps run on Sonnet 5.5, which
  can't read Opus's cache, and vice versa. Each executor conversation stays on one model, so no conversation
  is invalidated midway. Even so, the cache-reuse pattern changes. Run this as its own experiment, separate
  from the Jev-only comparison.
- **Run order and timing.** The harness alternates option order on each repeat. Rate-limit retries are hidden
  inside the SDK, which retries twice by default; they inflate latency but not tokens.
- **Small samples and path dependence.** One task and a few runs measure only this task. Decisions change the
  trajectory, so variance comes both from the model and from the decision path.
- **Thinking tokens** are billed as Claude output. They depend on the effort level, which is set in
  `EFFORT`. Changing effort shifts Option 1's decision cost a lot, so compare at the effort levels you would
  actually run in production.
- **Token counts aren't comparable across providers.** Jev and Claude use different tokenizers. Compare
  dollars, not tokens, between models.
- **Jev billing.** Unless the response includes a cost, Jev's cost is token count × the table price. Check it
  against your TypeSafe console.
- **Refusal fallbacks.** If Opus 5.5 declines a request, another model serves the turn at its own price. These
  calls are tagged `served-by:` in `calls.jsonl`. They should be rare on this task, but check for them.
- **Failed Jev calls aren't logged.** Network errors and 5xx responses fall back to Opus, and that Opus call
  is logged and costed. The failed Jev request isn't, so its latency is missing from the totals.
- **Held-out tests define "working".** Eleven tests cover the spec but can't prove the code correct. Read a
  sample of the generated code from each option in `runs/<timestamp>/`.
