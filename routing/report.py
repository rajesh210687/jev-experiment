"""Markdown comparison report from the per-run ledgers. Re-run on saved data with:

    python -m routing.report runs/<dir>/all_runs.json
"""

import json
import sys
from collections import Counter
from statistics import mean

from .config import PRICES, TIERS

CONFIG_ORDER = ["opt1-fine", "opt1-coarse", "opt2-fine", "opt2-coarse"]
NAMES = {"opt1-fine": "Opt 1 fine", "opt1-coarse": "Opt 1 coarse", "opt2-fine": "Opt 2 fine", "opt2-coarse": "Opt 2 coarse"}


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def usd(x: float) -> str:
    sign, x = ("-" if x < 0 else ""), abs(x)
    return f"{sign}${x:.4f}" if x >= 0.001 or x == 0 else f"{sign}${x:.6f}"


pct = lambda x: f"{x:+.1f}%"  # noqa: E731


def mmm(xs: list[float], f=usd) -> str:
    return f"{f(mean(xs))} ({f(min(xs))} to {f(max(xs))})" if xs else "n/a"


def cost(r: dict, **where) -> float:
    return sum(c["cost_usd"] for c in r["calls"] if all(c[k] == v for k, v in where.items()))


def per_run_mean(runs: list[dict], fn) -> float:
    return mean(fn(r) for r in runs) if runs else 0.0


def render(results: list[dict], settings: dict | None = None) -> str:
    by_cfg = {c: [r for r in results if r["config"] == c] for c in CONFIG_ORDER}
    by_cfg = {c: rs for c, rs in by_cfg.items() if rs}
    md: list[str] = ["# Routing experiment report\n"]

    md.append("## Prices used\n")
    md.append(table(["Model", "Input $/MTok", "Output $/MTok", "Cache write 5m", "Cache read", "Checked", "Source"],
                    [[k, p.input, p.output, p.cache_write_5m, p.cache_read, p.checked, p.source] for k, p in PRICES.items()]))
    if settings:
        md.append(f"\nSettings: {settings}\n")

    # --- headline ---------------------------------------------------------------------------
    md.append("\n## Total cost and latency per configuration (mean, min to max across runs)\n")
    rows = []
    for c, rs in by_cfg.items():
        ok = [r for r in rs if r["success"]]
        rows.append([NAMES[c], len(rs), f"{len(ok)}/{len(rs)}", mmm([cost(r) for r in rs]),
                     usd(mean(cost(r) for r in ok)) if ok else "n/a",
                     mmm([sum(x["latency_s"] for x in r["calls"]) for r in rs], lambda v: f"{v:.1f}s"),
                     mmm([r["wall_s"] for r in rs], lambda v: f"{v:.1f}s")])
    md.append(table(["Config", "Runs", "Succeeded", "Total cost", "Cost, successful runs only",
                     "Sum of API latency", "Wall clock"], rows))

    # --- tokens / calls / latency per model ---------------------------------------------------
    md.append("\n## Calls, tokens and latency per model (mean per run)\n")
    rows = []
    for c, rs in by_cfg.items():
        for t in TIERS + ["jev"]:
            f = lambda k, t=t, rs=rs: per_run_mean(rs, lambda r: sum(x[k] for x in r["calls"] if x["tier"] == t))  # noqa: E731
            n = per_run_mean(rs, lambda r, t=t: sum(1 for x in r["calls"] if x["tier"] == t))
            if n:
                rows.append([NAMES[c], t, f"{n:.1f}", f"{f('input_tokens'):.0f}", f"{f('output_tokens'):.0f}",
                             f"{f('cache_read_tokens'):.0f}/{f('cache_write_tokens'):.0f}", f"{f('latency_s'):.1f}s",
                             usd(f("cost_usd"))])
    md.append(table(["Config", "Model", "Calls", "Input tok", "Output tok", "Cache read/write tok", "Latency", "Cost"], rows))

    # --- cost per step ----------------------------------------------------------------------
    for gran in ("fine", "coarse"):
        o1, o2 = by_cfg.get(f"opt1-{gran}", []), by_cfg.get(f"opt2-{gran}", [])
        if not (o1 or o2):
            continue
        keys = list(dict.fromkeys(s["step"] for r in o1 + o2 for s in r["steps"]))
        md.append(f"\n## Cost per step, {gran} granularity (mean per run)\n")
        rows = []
        for k in keys:
            typ = next(s["step_type"] for r in o1 + o2 for s in r["steps"] if s["step"] == k)
            rows.append([f"{k} ({typ})", usd(per_run_mean(o1, lambda r: cost(r, step=k, kind="model"))),
                         usd(per_run_mean(o2, lambda r: cost(r, step=k, kind="model"))),
                         usd(per_run_mean(o2, lambda r: cost(r, step=k, kind="router")))])
        rows.append(["**Total**", usd(per_run_mean(o1, cost)), usd(per_run_mean(o2, lambda r: cost(r, kind="model"))),
                     usd(per_run_mean(o2, lambda r: cost(r, kind="router")))])
        md.append(table(["Step", "Opt 1 (Opus)", "Opt 2 Claude calls", "Opt 2 Jev routing"], rows))

    # --- routing distribution -----------------------------------------------------------------
    for gran in ("fine", "coarse"):
        rs = by_cfg.get(f"opt2-{gran}", [])
        if not rs:
            continue
        steps = [s for r in rs for s in r["steps"]]
        groups: dict[str, list[dict]] = {}
        for k in dict.fromkeys(s["step"] for s in steps):
            groups[k] = [s for s in steps if s["step"] == k]
        groups["codegen (all)"] = [s for s in steps if s["step_type"] == "codegen"]
        groups["ALL steps"] = steps

        def dist(ss: list[dict], field: str) -> str:
            c = Counter(s[field] for s in ss if s[field])
            n = sum(c.values()) or 1
            return " / ".join(f"{100 * c[t] // n}" for t in TIERS)

        md.append(f"\n## Routing distribution, Option 2 {gran} (% of steps: haiku / sonnet / opus)\n")
        md.append(table(["Step", "Steps", "Jev's raw pick", "Ran first (after confidence gate)", "Accepted output from"],
                        [[k, len(ss), dist(ss, "jev_choice"), dist(ss, "first_tier"), dist(ss, "final_tier")]
                         for k, ss in groups.items()]))

    # --- escalations and overhead -------------------------------------------------------------
    md.append("\n## Escalations, retries and Jev overhead (mean per run)\n")
    rows = []
    for c, rs in by_cfg.items():
        steps = lambda r: r["steps"]  # noqa: E731
        total = per_run_mean(rs, cost)
        jev = per_run_mean(rs, lambda r: cost(r, kind="router"))
        rows.append([NAMES[c],
                     f"{per_run_mean(rs, lambda r: sum(len(s['escalations']) for s in steps(r))):.2f}",
                     usd(per_run_mean(rs, lambda r: sum(s["escalation_cost_usd"] for s in steps(r)))),
                     f"{per_run_mean(rs, lambda r: sum(s['same_tier_retries'] for s in steps(r))):.2f}",
                     f"{per_run_mean(rs, lambda r: sum(s['low_confidence_bump'] for s in steps(r))):.2f}",
                     f"{per_run_mean(rs, lambda r: sum(1 for s in steps(r) if r['option'] == 2 and s['jev_choice'] is None)):.2f}",
                     usd(jev), f"{100 * jev / total:.1f}%" if total and jev else "-"])
    md.append(table(["Config", "Escalations", "Cost added by escalations*", "Same-model retries",
                     "Low-confidence bumps", "Jev errors", "Jev routing cost", "Jev share of total cost"], rows))
    md.append("\n*Spend on attempts that failed and were then escalated; the retry itself is in the totals. "
              "Option 1 has no higher tier, so its failures show up as same-model retries.")

    # --- success ------------------------------------------------------------------------------
    md.append("\n## Did each run produce working code? (held-out acceptance tests)\n")
    md.append(table(["Config", "Per run (acceptance passed/total, own tests passed/total)"],
                    [[NAMES[c], "; ".join(f"run{r['run']}: {'PASS' if r['success'] else 'FAIL'} "
                                          f"{r['acceptance'][0]}/{r['acceptance'][1]}, {r['own_tests'][0]}/{r['own_tests'][1]}"
                                          + (f" [{r['error'][:60]}]" if r["error"] else "") for r in rs)]
                     for c, rs in by_cfg.items()]))

    # --- savings ------------------------------------------------------------------------------
    md.append("\n## Option 2 vs Option 1 (negative = Option 2 is cheaper)\n")
    rows, totals = [], {}
    for gran in ("fine", "coarse"):
        a, b = by_cfg.get(f"opt1-{gran}"), by_cfg.get(f"opt2-{gran}")
        if not (a and b):
            continue
        ca, cb = mean(cost(r) for r in a), mean(cost(r) for r in b)
        totals[gran] = cb
        sa, sb = [cost(r) for r in a if r["success"]], [cost(r) for r in b if r["success"]]
        ok = f"{usd(mean(sb) - mean(sa))} ({pct(100 * (mean(sb) - mean(sa)) / mean(sa))})" if sa and sb else "n/a"
        rows.append([gran, usd(ca), usd(cb), usd(cb - ca), pct(100 * (cb - ca) / ca), ok])
    md.append(table(["Granularity", "Opt 1 mean", "Opt 2 mean", "Difference", "Difference %", "Difference, successful runs only"], rows))
    if len(totals) == 2:
        best = min(totals, key=totals.get)
        md.append(f"\n**Option 2 is cheaper at {best} granularity** ({usd(totals[best])} vs "
                  f"{usd(max(totals.values()))}) once Jev overhead and escalations are included.")
    return "\n".join(md)


if __name__ == "__main__":
    print(render(json.load(open(sys.argv[1]))))
