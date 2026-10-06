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
