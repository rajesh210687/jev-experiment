"""Turn the call ledger and quality results into a markdown comparison report."""

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from .config import JEV, PRICES
from .ledger import CallRecord
from .quality import Quality

STEPS = ["plan", "context", "route", "execute", "tests", "recovery", "fix"]
LABEL = {"option1": "Option 1 (Opus only)", "option2": "Option 2 (Opus + Jev)"}


@dataclass
class RunResult:
    option: str
    run: int
    quality: Quality | None
    wall_s: float
    error: str = ""


def _fmt_range(vals: list[float], fmt: str) -> str:
    m, lo, hi = (format(v, fmt) for v in (mean(vals), min(vals), max(vals)))
    return m if lo == hi else f"{m} ({lo}–{hi})"


def _delta(a: float, b: float) -> str:
    return "—" if a == 0 else f"{(b - a) / a:+.0%}"


def render(records: list[CallRecord], results: list[RunResult]) -> str:
    options = sorted({r.option for r in results})
    runs = {o: sorted({r.run for r in results if r.option == o}) for o in options}

    def per_run(o: str, f) -> list[float]:
        return [sum(f(c) for c in records if c.option == o and c.run == r) for r in runs[o]]

    is_opus = lambda c: c.model != JEV  # noqa: E731
    rows: list[tuple[str, dict[str, list[float]], str]] = [
        ("Total cost per run (USD)", {o: per_run(o, lambda c: c.cost_usd) for o in options}, ".4f"),
        ("Claude calls per run", {o: per_run(o, lambda c: is_opus(c)) for o in options}, ".1f"),
        ("Jev calls per run", {o: per_run(o, lambda c: not is_opus(c)) for o in options}, ".1f"),
        ("Opus fallback calls (Jev unsure or unavailable)", {o: per_run(o, lambda c: c.note == "jev-fallback") for o in options}, ".1f"),
        ("Claude input tokens (uncached)", {o: per_run(o, lambda c: c.input_tokens * is_opus(c)) for o in options}, ",.0f"),
        ("Claude cache-read tokens", {o: per_run(o, lambda c: c.cache_read_tokens) for o in options}, ",.0f"),
        ("Claude cache-write tokens", {o: per_run(o, lambda c: c.cache_write_tokens) for o in options}, ",.0f"),
        ("Claude output tokens (incl. thinking)", {o: per_run(o, lambda c: c.output_tokens * is_opus(c)) for o in options}, ",.0f"),
        ("Jev input tokens", {o: per_run(o, lambda c: c.input_tokens * (not is_opus(c))) for o in options}, ",.0f"),
        ("Model latency per run (s, summed)", {o: per_run(o, lambda c: c.latency_s) for o in options}, ".1f"),
        ("Wall-clock per run (s)", {o: [r.wall_s for r in results if r.option == o] for o in options}, ".1f"),
    ]

    out = ["# Cost comparison\n"]
    n_runs = {o: len(runs[o]) for o in options}
    out.append(f"Runs per option: {', '.join(f'{LABEL[o]}: {n_runs[o]}' for o in options)}. "
               "Values are means per run; ranges in parentheses when runs > 1.\n")

    header = "| Metric | " + " | ".join(LABEL[o] for o in options) + (" | Δ (2 vs 1) |" if len(options) == 2 else " |")
    out += [header, "|" + "---|" * (header.count("|") - 1)]

    def quality_cell(o: str) -> str:
        rs = [r for r in results if r.option == o]
        ok = sum(1 for r in rs if r.quality and r.quality.succeeded)
        acc = [f"{r.quality.acceptance_passed}/{r.quality.acceptance_total}" if r.quality else "error" for r in rs]
        return f"{'✅' if ok == len(rs) else '❌'} {ok}/{len(rs)} runs (acceptance {', '.join(acc)})"

    def own_cell(o: str) -> str:
        return ", ".join(f"{r.quality.own_passed}/{r.quality.own_total}" if r.quality else "error"
                         for r in results if r.option == o)

    extra = " | |" if len(options) == 2 else " |"
    out.append("| Task succeeded (held-out acceptance tests) | " + " | ".join(quality_cell(o) for o in options) + extra)
    out.append("| Agent's own tests passing | " + " | ".join(own_cell(o) for o in options) + extra)
    for name, vals, fmt in rows:
        cells = [_fmt_range(vals[o], fmt) for o in options]
        d = f" | {_delta(mean(vals[options[0]]), mean(vals[options[1]]))} |" if len(options) == 2 else " |"
        out.append(f"| {name} | " + " | ".join(cells) + d)

    jev_calls = [c for c in records if c.model == JEV]
    if jev_calls:
        asked = sum(c.questions for c in jev_calls)
        low = sum(c.low_confidence for c in jev_calls)
        out.append(f"\nJev answered {asked} questions; {asked - low} accepted ({(asked - low) / asked:.0%}), "
                   f"{low} below threshold and re-asked to Opus.\n")

    # Cost per step type
    out.append("\n## Cost per step type (mean per run)\n")
    out.append("| Step | " + " | ".join(f"{LABEL[o]} cost | calls (Claude/Jev) | latency s" for o in options) + " |")
    out.append("|---|" + "---|---|---|" * len(options))
    for step in STEPS:
        cells = []
        for o in options:
            cs = [c for c in records if c.option == o and c.step == step]
            n = n_runs[o]
            cells.append(f"${sum(c.cost_usd for c in cs) / n:.4f} | "
                         f"{sum(is_opus(c) for c in cs) / n:.1f}/{sum(not is_opus(c) for c in cs) / n:.1f} | "
                         f"{sum(c.latency_s for c in cs) / n:.1f}")
        out.append(f"| {step} | " + " | ".join(cells) + " |")

    # Per model per step detail (totals across runs)
    out.append("\n## Detail: per option × step × model (totals across runs)\n")
    out.append("| Option | Step | Model | Calls | Input | Cache read | Cache write | Output | Latency s | Cost USD |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    agg: dict[tuple, list[CallRecord]] = defaultdict(list)
    for c in records:
        agg[(c.option, c.step, c.model)].append(c)
    for (o, step, model), cs in sorted(agg.items(), key=lambda kv: (kv[0][0], STEPS.index(kv[0][1]), kv[0][2])):
        out.append(f"| {o} | {step} | {model} | {len(cs)} | {sum(c.input_tokens for c in cs):,} | "
                   f"{sum(c.cache_read_tokens for c in cs):,} | {sum(c.cache_write_tokens for c in cs):,} | "
                   f"{sum(c.output_tokens for c in cs):,} | {sum(c.latency_s for c in cs):.1f} | "
                   f"{sum(c.cost_usd for c in cs):.4f} |")

    # Prices used
    used = sorted({c.model for c in records})
    out.append("\n## Prices used (USD per million tokens)\n")
    out.append("| Model | Input | Cache write (5m) | Cache read | Output | Source | Checked |")
    out.append("|---|---|---|---|---|---|---|")
    for m in used:
        p = PRICES[m]
        out.append(f"| {m} | {p.input} | {p.cache_write_5m} | {p.cache_read} | {p.output} | {p.source} | {p.checked} |")
    if any(c.note == "cost-estimated-from-price-table" for c in jev_calls):
        out.append("\nSome Jev calls had no `usage.cost` in the response; those were priced from the table above.")
    elif jev_calls:
        out.append("\nJev costs are the `usage.cost` values the Jev API reported per call.")

    errors = [r for r in results if r.error]
    if errors:
        out.append("\n## Errors\n")
        out += [f"- {r.option} run {r.run}: {r.error}" for r in errors]
    return "\n".join(out) + "\n"
