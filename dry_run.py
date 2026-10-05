"""Dry run: stub both APIs (no keys, no spend) and drive the real harness end to end.

The fake Opus writes a known-good solution plus one broken test, so the recovery path runs too.
Usage: python dry_run.py [--repeats 2]
"""
import json, os, runpy, sys
from pathlib import Path
from types import SimpleNamespace as NS
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ANTHROPIC_API_KEY", "x"); os.environ.setdefault("TYPESAFE_API_KEY", "x")
from harness import llm

IMPL = {
"orders_dedupe/__init__.py": "",
"orders_dedupe/__main__.py": '''import argparse, csv, sys
from pathlib import Path
REQ = ["order_id", "customer_email", "sku", "quantity", "order_date"]
def flag(rows):
    by_id, by_key, out = {}, {}, []
    for r in rows:
        oid = r["order_id"].strip()
        try: q = int(r["quantity"].strip())
        except ValueError: q = r["quantity"].strip()
        key = (r["customer_email"].strip().lower(), r["sku"].strip().lower(), q, r["order_date"].strip())
        orig = by_id.get(oid) or by_key.get(key)
        if orig is None:
            by_id.setdefault(oid, oid); by_key.setdefault(key, oid)
        else:
            by_id.setdefault(oid, orig); by_key.setdefault(key, orig)
        out.append({**r, "is_duplicate": "true" if orig else "false", "duplicate_of": orig or ""})
    return out
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("--output")
    a = ap.parse_args(argv)
    p = Path(a.input)
    if not p.exists():
        print(f"error: {p} not found", file=sys.stderr); return 2
    with p.open(newline="") as f:
        rd = csv.DictReader(f); fields = rd.fieldnames or []
        miss = [c for c in REQ if c not in fields]
        if miss:
            print(f"error: missing column: {', '.join(miss)}", file=sys.stderr); return 2
        out = flag(list(rd))
    dest = open(a.output, "w", newline="") if a.output else sys.stdout
    w = csv.DictWriter(dest, fieldnames=fields + ["is_duplicate", "duplicate_of"]); w.writeheader(); w.writerows(out)
    if a.output: dest.close()
    print(f"{len(out)} rows, {sum(o['is_duplicate'] == 'true' for o in out)} duplicates", file=sys.stderr)
    return 0
raise SystemExit(main())
''',
"tests/test_cli.py": '''import subprocess, sys
def test_runs(tmp_path):
    f = tmp_path / "o.csv"; f.write_text("order_id,customer_email,sku,quantity,order_date\\n1,a,b,1,d\\n1,a,b,1,d\\n")
    r = subprocess.run([sys.executable, "-m", "orders_dedupe", str(f)], capture_output=True, text=True)
    assert "2 rows, 1 duplicates" in r.stderr
''',
"tests/test_rules.py": '''def test_placeholder():
    assert BROKEN  # first version fails, forcing the recovery path
''',
}

def usage(i, o): return NS(input_tokens=i, output_tokens=o, cache_creation_input_tokens=100, cache_read_input_tokens=50)
def text(t): return NS(type="text", text=t)

def fake_send(self, **kw):
    model = kw["model"]
    fmt = kw["output_config"].get("format")
    if fmt:
        props = fmt["schema"]["properties"]
        if "subtasks" in props:
            body = {"subtasks": [{"title": "Package and CLI", "detail": "x"}, {"title": "Tests", "detail": "y"}]}
        else:
            body = {k: (True if v["type"] == "boolean" else v["enum"][0]) for k, v in props.items()}
        return NS(content=[NS(type="thinking", thinking=""), text(json.dumps(body))], stop_reason="end_turn",
                  usage=usage(800, 60), model=model, stop_details=None)
    last = kw["messages"][-1]
    if isinstance(last["content"], list):  # tool results came back
        return NS(content=[text("done")], stop_reason="end_turn", usage=usage(3000, 20), model=model, stop_details=None)
    files = dict(IMPL)
    if "Recovery path" in last["content"]:
        files = {"tests/test_rules.py": "def test_ok():\n    assert True\n"}
    blocks = [NS(type="tool_use", id=f"t{i}", name="write_file", input={"path": p, "content": c})
              for i, (p, c) in enumerate(files.items())]
    return NS(content=blocks, stop_reason="tool_use", usage=usage(2500, 1500), model=model, stop_details=None)

def fake_post(self, payload):
    ans = {}
    for k, q in payload["questions"].items():
        if q["type"] == "noul": ans[k] = {"type": "noul", "noul": 0.95 if k.endswith("0") else 0.6}
        elif q["type"] == "choice": ans[k] = {"type": "choice", "choice": next(iter(q["criteria"])), "confidence": 0.9}
        else: ans[k] = {"type": "score", "score": 1.4, "confidence": 0.55}
    return {"answers": ans, "usage": {"input_tokens": 400, "output_tokens": 0, "cost": 400 * 0.042 / 1e6}}

llm.Opus._send = fake_send
llm.Jev._post = fake_post
sys.argv = ["run.py", *sys.argv[1:]]
runpy.run_path(str(ROOT / "run.py"), run_name="__main__")
