"""Dry run: stub Claude and Jev (no keys, no spend) and drive the real harness end to end.

The fake models write a known-good solution, with scripted failures so every path runs:
  - Haiku returns prose instead of files on "core"                      -> malformed -> escalate
  - Sonnet renames a public function on "edge_cases"                    -> inconsistent -> escalate
  - Haiku/Sonnet write a failing test on "unit_tests"                   -> fix step
  - Haiku "fixes" nothing on "fix_errors"                               -> tests still fail -> escalate
Numbers it prints are FAKE. Usage: python -m routing.dry_run [--runs 2]
"""

import json
import os
import sys
import time
from types import SimpleNamespace as NS

os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("TYPESAFE_API_KEY", "x")

from . import clients, runner  # noqa: E402
from .config import MODEL_IDS  # noqa: E402
from .steps import LAYOUT  # noqa: E402

TIER_OF = {v: k for k, v in MODEL_IDS.items()}

MODELS = '''from dataclasses import dataclass

REQUIRED = ("order_id", "customer_email", "sku", "quantity", "order_date")


@dataclass(frozen=True)
class Flagged:
    row: dict
    is_duplicate: bool
    duplicate_of: str
'''
CORE_V1 = '''from .models import Flagged


def flag_duplicates(rows):
    seen, out = {}, []
    for r in rows:
        oid = r["order_id"]
        out.append(Flagged(r, oid in seen, seen.get(oid, "")))
        seen.setdefault(oid, oid)
    return out
'''
CORE_V2 = '''from .models import Flagged


def _key(r):
    q = r["quantity"].strip()
    try:
        q = int(q)
    except ValueError:
        pass
    return (r["customer_email"].strip().lower(), r["sku"].strip().lower(), q, r["order_date"].strip())


def flag_duplicates(rows):
    by_id, by_key, out = {}, {}, []
    for r in rows:
        oid, key = r["order_id"].strip(), _key(r)
        orig = by_id.get(oid) or by_key.get(key)
        by_id.setdefault(oid, orig or oid)
        by_key.setdefault(key, orig or oid)
        out.append(Flagged(r, bool(orig), orig or ""))
    return out
'''
CORE_RENAMED = CORE_V2.replace("flag_duplicates", "flag_all")
CLI = '''import argparse
import csv
import sys
from pathlib import Path

from .core import flag_duplicates
from .models import REQUIRED


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orders_dedupe")
    ap.add_argument("input")
    ap.add_argument("--output")
    a = ap.parse_args(argv)
    p = Path(a.input)
    if not p.exists():
        print(f"error: {p} not found", file=sys.stderr)
        return 2
    with p.open(newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        miss = [c for c in REQUIRED if c not in fields]
        if miss:
            print(f"error: missing column: {', '.join(miss)}", file=sys.stderr)
            return 2
        flagged = flag_duplicates(list(rd))
    dest = open(a.output, "w", newline="") if a.output else sys.stdout
    w = csv.DictWriter(dest, fieldnames=fields + ["is_duplicate", "duplicate_of"])
    w.writeheader()
    for fl in flagged:
        w.writerow({**fl.row, "is_duplicate": "true" if fl.is_duplicate else "false", "duplicate_of": fl.duplicate_of})
    if a.output:
        dest.close()
    print(f"{len(flagged)} rows, {sum(f.is_duplicate for f in flagged)} duplicates", file=sys.stderr)
    return 0
'''
FILES = {
    "orders_dedupe/__init__.py": '"""Orders dedupe."""\n__version__ = "0.1.0"\n',
    "orders_dedupe/__main__.py": "import sys\n\nfrom .cli import main\n\nsys.exit(main())\n",
    "orders_dedupe/models.py": MODELS,
    "orders_dedupe/cli.py": CLI,
    "tests/test_core.py": "from orders_dedupe.core import flag_duplicates\n\n\ndef test_id_dup():\n"
                          "    rows = [{'order_id': '1', 'customer_email': 'a', 'sku': 's', 'quantity': '1', 'order_date': 'd'}] * 2\n"
                          "    assert [f.is_duplicate for f in flag_duplicates(rows)] == [False, True]\n",
    "tests/test_cli.py": "import subprocess\nimport sys\n\n\ndef test_missing_file(tmp_path):\n"
                         "    r = subprocess.run([sys.executable, '-m', 'orders_dedupe', str(tmp_path / 'nope.csv')], capture_output=True, text=True)\n"
                         "    assert r.returncode == 2\n",
    "README.md": "# orders_dedupe\n\nUsage: `python -m orders_dedupe INPUT_CSV [--output OUT]`\n",
}
BROKEN_TEST = FILES["tests/test_core.py"].replace("[False, True]", "[True, True]")
PLAN = {"modules": {p: {"purpose": "x", "api": []} for p in LAYOUT}, "edge_cases": ["whitespace", "case"],
        "test_cases": ["id dup", "content dup"], "cli_contract": "argv: INPUT [--output]; exit 2 on bad input"}


def fake_send(self, **kw):
    tier, body = TIER_OF[kw["model"]], kw["messages"][0]["content"][1]["text"]
    time.sleep({"haiku": 0.02, "sonnet": 0.04, "opus": 0.08}[tier])  # fake latency, in ms-scale
    if "Design the solution" in body:
        text = json.dumps(PLAN)
    elif "Choose which test files" in body:
        text = json.dumps({"targets": [ln.split(" ")[0] for ln in body.split("<candidates>")[1].split("</candidates>")[0].split("\n") if ln.strip()]})
    elif "The tests below fail" in body:
        text = "no idea" if tier == "haiku" else f'<file path="tests/test_core.py">\n{FILES["tests/test_core.py"]}</file>'
    else:
        want = body.split("Write exactly these files: ")[1].split("\n")[0].split(", ")
        key_core_v2 = "edge" in body.split("<step>")[1][:400] or "Harden" in body
        files = {p: FILES.get(p, "") for p in want}
        if "orders_dedupe/cli.py" not in want and "orders_dedupe/__main__.py" in want:
            files["orders_dedupe/__main__.py"] = "import sys\n\nsys.exit(0)  # wired up in the CLI step\n"
        if "orders_dedupe/core.py" in want:
            files["orders_dedupe/core.py"] = CORE_V2 if key_core_v2 else CORE_V1
            if key_core_v2 and "Part 1" not in body and tier == "sonnet":
                files["orders_dedupe/core.py"] = CORE_RENAMED  # scripted inconsistency
            if not key_core_v2 and tier == "haiku":
                files = {}  # scripted malformed output
        if "tests/test_core.py" in want and tier != "opus":
            files["tests/test_core.py"] = BROKEN_TEST
        text = "Sure, here is the code." if not files else "\n".join(f'<file path="{p}">\n{c}</file>' for p, c in files.items())
    u = NS(input_tokens=len(body) // 4 + 400, output_tokens=len(text) // 4 + 200,
           cache_creation_input_tokens=0, cache_read_input_tokens=0)
    return NS(content=[NS(type="text", text=text)], stop_reason="end_turn", usage=u, model=kw["model"])


PICKS = {"plan": ("opus", .9), "scaffold": ("haiku", .95), "models": ("haiku", .8), "core": ("haiku", .85),
         "edge_cases": ("haiku", .5), "cli": ("sonnet", .9), "unit_tests": ("haiku", .9), "docs_cleanup": ("haiku", .9),
         "select_tests": ("haiku", .95), "fix_errors": ("haiku", .8), "scaffold_models": ("haiku", .9),
         "core_edge_cases": ("sonnet", .8), "cli_unit_tests": ("sonnet", .8)}


def fake_post(self, payload):
    time.sleep(0.01)
    state = json.loads(payload["state"])
    choice, conf = PICKS[state["step_key"]]
    return {"answers": {"tier": {"choice": choice, "confidence": conf}},
            "usage": {"input_tokens": len(payload["state"]) // 4 + 250, "output_tokens": 0}}


clients.Claude.__init__ = lambda self, ledger: setattr(self, "ledger", ledger)
clients.Jev.__init__ = lambda self, ledger: setattr(self, "ledger", ledger)
clients.Claude._send = fake_send
clients.Jev._post = fake_post

if __name__ == "__main__":
    sys.argv = ["run_routing.py", *sys.argv[1:]]
    runner.main()
