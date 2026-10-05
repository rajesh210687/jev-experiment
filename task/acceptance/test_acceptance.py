"""Held-out acceptance tests. The agent never sees these; the harness runs them after each run.

The harness sets HARNESS_WORKSPACE to the agent's project directory.
"""

import csv
import io
import os
import subprocess
import sys
from pathlib import Path

WS = Path(os.environ["HARNESS_WORKSPACE"]).resolve()
HEADER = "order_id,customer_email,sku,quantity,order_date,channel\n"


def run_cli(tmp_path: Path, text: str, *args: str) -> subprocess.CompletedProcess:
    src = tmp_path / "orders.csv"
    src.write_text(text)
    return subprocess.run([sys.executable, "-m", "orders_dedupe", str(src), *args],
                          cwd=WS, capture_output=True, text=True, timeout=60)


def rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def test_no_duplicates(tmp_path):
    r = run_cli(tmp_path, HEADER + "1,a@x.com,SKU1,1,2026-01-01,web\n2,b@x.com,SKU1,1,2026-01-01,web\n")
    assert r.returncode == 0, r.stderr
    out = rows(r.stdout)
    assert [o["is_duplicate"] for o in out] == ["false", "false"]
    assert [o["duplicate_of"] for o in out] == ["", ""]


def test_duplicate_order_id_with_whitespace(tmp_path):
    r = run_cli(tmp_path, HEADER + "1001,a@x.com,SKU1,1,2026-01-01,web\n 1001 ,z@x.com,SKU9,5,2026-02-02,store\n")
    assert r.returncode == 0, r.stderr
    out = rows(r.stdout)
    assert out[0]["is_duplicate"] == "false"
    assert out[1]["is_duplicate"] == "true"
    assert out[1]["duplicate_of"] == "1001"


def test_duplicate_by_content_normalized(tmp_path):
    r = run_cli(tmp_path, HEADER
                + "1,Alice@Example.com,sku-7,2,2026-03-01,web\n"
                + "2, alice@example.COM ,SKU-7,02,2026-03-01,app\n")
    assert r.returncode == 0, r.stderr
    out = rows(r.stdout)
    assert out[1]["is_duplicate"] == "true"
    assert out[1]["duplicate_of"] == "1"


def test_different_quantity_is_not_duplicate(tmp_path):
    r = run_cli(tmp_path, HEADER + "1,a@x.com,SKU1,1,2026-01-01,web\n2,a@x.com,SKU1,3,2026-01-01,web\n")
    assert [o["is_duplicate"] for o in rows(r.stdout)] == ["false", "false"]


def test_points_to_earliest_match(tmp_path):
    r = run_cli(tmp_path, HEADER
                + "1,a@x.com,SKU1,1,2026-01-01,web\n"
                + "2,a@x.com,SKU1,1,2026-01-01,web\n"
                + "3,a@x.com,SKU1,1,2026-01-01,web\n")
    out = rows(r.stdout)
    assert [o["is_duplicate"] for o in out] == ["false", "true", "true"]
    assert [o["duplicate_of"] for o in out] == ["", "1", "1"]


def test_preserves_columns_and_appends_new_ones(tmp_path):
    r = run_cli(tmp_path, HEADER + "1,a@x.com,SKU1,1,2026-01-01,web\n")
    header = next(csv.reader(io.StringIO(r.stdout)))
    assert header == ["order_id", "customer_email", "sku", "quantity", "order_date", "channel",
                      "is_duplicate", "duplicate_of"]
    assert rows(r.stdout)[0]["channel"] == "web"


def test_summary_on_stderr(tmp_path):
    r = run_cli(tmp_path, HEADER + "1,a@x.com,SKU1,1,2026-01-01,web\n1,a@x.com,SKU1,1,2026-01-01,web\n")
    assert "2 rows, 1 duplicates" in r.stderr


def test_output_option_writes_file(tmp_path):
    dest = tmp_path / "out.csv"
    r = run_cli(tmp_path, HEADER + "1,a@x.com,SKU1,1,2026-01-01,web\n", "--output", str(dest))
    assert r.returncode == 0, r.stderr
    assert rows(dest.read_text())[0]["is_duplicate"] == "false"


def test_missing_required_column(tmp_path):
    r = run_cli(tmp_path, "order_id,customer_email,sku,order_date\n1,a@x.com,SKU1,2026-01-01\n")
    assert r.returncode == 2
    assert "missing column" in r.stderr.lower()
    assert "quantity" in r.stderr


def test_missing_file():
    r = subprocess.run([sys.executable, "-m", "orders_dedupe", str(WS / "does-not-exist.csv")],
                       cwd=WS, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert r.stderr.strip()


def test_header_only(tmp_path):
    r = run_cli(tmp_path, HEADER)
    assert r.returncode == 0, r.stderr
    assert rows(r.stdout) == []
    assert "0 rows, 0 duplicates" in r.stderr
