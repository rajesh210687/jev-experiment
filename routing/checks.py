"""Parsing and validation of model output. A failed check is what triggers escalation."""

import ast
import json
import re
import sys

from .workspace import Workspace

PKG = "orders_dedupe"
FILE_RE = re.compile(r'<file path="([^"]+)">\n?(.*?)\n?</file>', re.S)


def parse_files(text: str) -> dict[str, str]:
    return {p.strip(): body + "\n" for p, body in FILE_RE.findall(text)}


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        v = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) else None


def safe_path(p: str) -> bool:
    return not p.startswith("/") and ".." not in p.split("/")


# --- AST helpers: top-level names, API digests ---------------------------------------------

def top_level_names(src: str, include_private: bool = True) -> set[str]:
    names: set[str] = set()
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return names if include_private else {x for x in names if not x.startswith("_")}


def _sig(n: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    ret = f" -> {ast.unparse(n.returns)}" if n.returns else ""
    return f"def {n.name}({ast.unparse(n.args)}){ret}"


def digest(src: str) -> str:
    """Signatures only: what a later step needs to stay consistent, at a fraction of the tokens."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "(unparseable)"
    out: list[str] = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_sig(n))
        elif isinstance(n, ast.ClassDef):
            out += [f"@{ast.unparse(d)}" for d in n.decorator_list]
            out.append(f"class {n.name}:")
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append("    " + _sig(m))
                elif isinstance(m, ast.AnnAssign):
                    out.append("    " + ast.unparse(m))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and len(ast.unparse(n)) < 120:
            out.append(ast.unparse(n))
    return "\n".join(out) or "(no public definitions)"


def error_type(output: str) -> str:
    for pat, name in [(r"ModuleNotFoundError|ImportError", "ImportError"), (r"SyntaxError", "SyntaxError"),
                      (r"AttributeError", "AttributeError"), (r"TypeError", "TypeError"),
                      (r"AssertionError|\bassert ", "AssertionError"), (r"timed out", "Timeout")]:
        if re.search(pat, output):
            return name
    return "Other"


# --- Validation ------------------------------------------------------------------------------

def check_internal_imports(ws: Workspace) -> str | None:
    """Every `from orders_dedupe.x import y` must resolve to a module and name that exist now.

    This is the check that catches a later sub-step contradicting an earlier one.
    """
    files = ws.files()
    for path in files:
        if not path.endswith(".py") or not (path.startswith(PKG + "/") or path.startswith("tests/")):
            continue
        try:
            tree = ast.parse(ws.read(path))
        except SyntaxError as e:
            return f"{path}: SyntaxError: {e}"
        for n in ast.walk(tree):
            if not isinstance(n, ast.ImportFrom):
                continue
            mod = PKG + (f".{n.module}" if n.module else "") if n.level else (n.module or "")
            if mod != PKG and not mod.startswith(PKG + "."):
                continue
            target = mod.replace(".", "/")
            src_path = next((c for c in (f"{target}.py", f"{target}/__init__.py") if c in files), None)
            if src_path is None:
                return f"{path}: imports module {mod}, which does not exist"
            names = top_level_names(ws.read(src_path))
            for a in n.names:
                if a.name != "*" and a.name not in names and f"{target}/{a.name}.py" not in files:
                    return f"{path}: imports {a.name} from {mod}, which does not define it"
    return None


def validate(ws: Workspace, new: dict[str, str], before: dict[str, str | None]) -> str | None:
    """Run after the files are written. Returns a failure reason, or None if all checks pass."""
    for p, src in new.items():  # 1. syntax
        if p.endswith(".py"):
            try:
                ast.parse(src)
            except SyntaxError as e:
                return f"{p}: SyntaxError: {e.msg} (line {e.lineno})"
    for p, src in new.items():  # 2. a rewritten file must keep the public names earlier steps relied on
        old = before.get(p)
        if old and p.startswith(PKG + "/") and p.endswith(".py"):
            lost = top_level_names(old, False) - top_level_names(src, False)
            if lost:
                return f"{p}: dropped public names {sorted(lost)} that earlier steps defined"
    bad = check_internal_imports(ws)  # 3. cross-file consistency
    if bad:
        return f"inconsistent with earlier code: {bad}"
    for p in new:  # 4. the code actually imports
        if p.startswith(PKG + "/") and p.endswith(".py") and not p.endswith("__main__.py"):
            mod = p[:-3].replace("/", ".").removesuffix(".__init__")
            r = ws.run([sys.executable, "-c", f"import {mod}"])
            if r.returncode:
                return f"import {mod} failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}"
    if f"{PKG}/__main__.py" in new:  # 5. the entry point starts
        r = ws.run([sys.executable, "-m", PKG, "--help"])
        if r.returncode:
            return f"python -m {PKG} --help exited {r.returncode}: {r.stderr.strip()[-200:]}"
    tests = [p for p in new if p.startswith("tests/") and p.endswith(".py")]
    if tests:  # 6. new tests are collectable (whether they PASS is the fix step's business)
        r = ws.pytest(["--collect-only", *tests])
        if not r.ok:
            return f"tests do not collect: {r.output.strip()[-300:]}"
    return None
