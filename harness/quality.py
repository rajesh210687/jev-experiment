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
