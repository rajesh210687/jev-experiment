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
