"""The fixed step sequence, shared by both options.

plan -> code generation (7 sub-steps "fine", or 4 groups "coarse") -> [select tests -> run -> fix]*

The file layout is fixed by the harness (so both options and all granularities build the same
thing); the plan step fills in the design: function signatures, edge cases, test cases.
"""

from dataclasses import dataclass

PKG = "orders_dedupe"
LAYOUT = [f"{PKG}/__init__.py", f"{PKG}/__main__.py", f"{PKG}/models.py", f"{PKG}/core.py",
          f"{PKG}/cli.py", "tests/test_core.py", "tests/test_cli.py", "README.md"]
ALL_PACKAGE_FILES = "*"  # in needs_full: every existing package file


@dataclass(frozen=True)
class StepSpec:
    key: str
    step_type: str  # plan | codegen | select_tests | fix_errors
    summary: str  # one line; this is what Jev reads
    brief: str  # the instruction the model reads
    writes: tuple[str, ...] = ()
    needs_full: tuple[str, ...] = ()  # existing files passed in full; other package files as signatures
    plan_keys: tuple[str, ...] = ()  # which parts of the plan this step needs
    lines: int = 0  # rough lines of output expected; a router signal
    may_edit_package: bool = False  # cleanup may rewrite any package file (public API must survive)


PLAN = StepSpec(
    "plan", "plan", "Design the module APIs, edge cases and test cases for the CSV duplicate flagger",
    "Design the solution. Return ONE JSON object (no prose) with keys: "
    '"modules" (an object with one entry per path in <project_layout>, each {"purpose": str, "api": '
    '[Python signatures, one per string]}), "edge_cases" (list of strings), "test_cases" (list of '
    'strings), "cli_contract" (string: argv, exit codes, stderr messages, output columns).',
    lines=60,
)

FINE = [
    StepSpec("scaffold", "codegen", "Create package skeleton files",
             f"Create the package skeleton. {PKG}/__init__.py holds only a docstring and "
             f'__version__ = "0.1.0". {PKG}/__main__.py is a placeholder that imports nothing from the '
             "package and exits 0 (it is wired to the CLI in a later step).",
             writes=(f"{PKG}/__init__.py", f"{PKG}/__main__.py"), lines=15),
    StepSpec("models", "codegen", "Define the data models",
             "Implement the data models listed in the plan for models.py. No I/O.",
             writes=(f"{PKG}/models.py",), plan_keys=(), lines=40),
    StepSpec("core", "codegen", "Implement the core duplicate-detection logic",
             "Implement the core duplicate-detection function(s) listed in the plan for core.py, using the "
             "models. Implement both matching rules; leave unusual-input handling to the next step.",
             writes=(f"{PKG}/core.py",), needs_full=(f"{PKG}/models.py",), lines=80),
    StepSpec("edge_cases", "codegen", "Harden core logic for edge cases",
             "Harden core.py for every item in edge_cases (whitespace, case, integer quantity "
             "normalization, empty input, chains of duplicates). Rewrite core.py in full; keep every "
             "public name it already defines.",
             writes=(f"{PKG}/core.py",), needs_full=(f"{PKG}/core.py", f"{PKG}/models.py"),
             plan_keys=("edge_cases",), lines=50),
    StepSpec("cli", "codegen", "Wire CSV input/output and the command-line interface",
             "Implement cli.py (argument parsing, CSV reading and writing, error handling, exit codes, "
             "summary line) per cli_contract, using the models and core. Rewrite __main__.py so that "
             "`python -m orders_dedupe` runs cli.main().",
             writes=(f"{PKG}/cli.py", f"{PKG}/__main__.py"), needs_full=(f"{PKG}/models.py",),
             plan_keys=("cli_contract",), lines=70),
    StepSpec("unit_tests", "codegen", "Write unit tests for core logic and CLI",
             "Write pytest tests: tests/test_core.py for the core logic, tests/test_cli.py for the CLI "
             "(run it with subprocess and `python -m orders_dedupe`). Cover every item in test_cases.",
             writes=("tests/test_core.py", "tests/test_cli.py"), plan_keys=("test_cases", "cli_contract"),
             lines=150),
    StepSpec("docs_cleanup", "codegen", "Write README and tidy code",
             "Write README.md (usage, examples, duplicate rules). You may also rewrite package files to "
             "remove dead code or improve docstrings, but behaviour and every public name must not change.",
             writes=("README.md",), needs_full=(ALL_PACKAGE_FILES,), may_edit_package=True, lines=60),
]


def _merge(specs: list[StepSpec]) -> StepSpec:
    uniq = lambda xs: tuple(dict.fromkeys(x for s in specs for x in xs(s)))  # noqa: E731
    return StepSpec(
        key="_".join(s.key for s in specs), step_type="codegen",
        summary=" + ".join(s.summary for s in specs),
        brief="\n".join(f"Part {i}: {s.brief}" for i, s in enumerate(specs, 1)),
        writes=uniq(lambda s: s.writes), needs_full=uniq(lambda s: s.needs_full),
        plan_keys=uniq(lambda s: s.plan_keys), lines=sum(s.lines for s in specs),
        may_edit_package=any(s.may_edit_package for s in specs),
    )


COARSE = [_merge(FINE[0:2]), _merge(FINE[2:4]), _merge(FINE[4:6]), FINE[6]]


def codegen_steps(granularity: str) -> list[StepSpec]:
    return {"fine": FINE, "coarse": COARSE}[granularity]


# The two steps whose inputs are produced at run time (so they have no static writes/needs).
SELECT_TESTS = StepSpec(
    "select_tests", "select_tests", "Choose which test files to run first",
    "Choose which test files to run first to get the fastest useful feedback on the files that "
    'changed. Return ONE JSON object (no prose): {"targets": ["tests/test_x.py", ...]}, using only '
    "paths from <candidates>.", lines=3)
FIX_ERRORS = StepSpec(
    "fix_errors", "fix_errors", "Fix failing tests or errors",
    "The tests below fail. Fix the root cause with the smallest change; edit the code or the test, "
    "whichever is actually wrong. Return every file you change in full.", lines=30)
