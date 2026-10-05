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
