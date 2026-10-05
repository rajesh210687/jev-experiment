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
