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
