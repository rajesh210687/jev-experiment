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
