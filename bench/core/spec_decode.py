"""Speculative-decoding recording and comparison for bench.

``bench run --speculative-config`` records vLLM's own acceptance counters at
three granularities, and ``bench validate`` puts them beside the simulator's:

* per tick, in ``timeseries.csv``, from ``SchedulerStats.spec_decoding_stats``
  (``vllm/v1/metrics/stats.py``), which the scheduler fills once per step from
  ``SpecDecodingStats.observe_draft`` (``vllm/v1/spec_decode/metrics.py``);
* per run, in ``meta.json`` under ``speculative_decoding``: the same counters
  summed, per draft position as well;
* per request, in ``requests.jsonl`` under ``spec_decode``: vLLM 0.31's
  ``per_request_spec_decode_metrics`` payload
  (``RequestSpecDecodeMetrics.to_dict``), carried on the finished output's
  ``CompletionOutput.spec_decode_metrics`` for single-sequence requests.

vLLM accepts a prefix of the draft, so ``accepted_per_pos[i]`` counts steps
that accepted positions 1..i+1. Divided by the verification steps it is
P(accepted >= i+1), the per-position rate vLLM's own logger reports
(``SpecDecodingLogging.log``) and the curve the simulator's
``--spec-position-acceptance`` takes. It cannot rise with position.

Kept free of vLLM imports so ``bench validate`` runs without vLLM installed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


# What makes vLLM draft a varying number of tokens per step
# (``vllm/config/speculative.py`` at v0.31.0): a batch-size schedule that picks
# N per batch (``uses_dynamic_speculative_decoding``), DSpark's per-request
# verification budget, and the proposers that draft only what they match --
# n-gram (none when nothing matches) and suffix decoding (N is a maximum). The
# simulator drafts a fixed N per caught-up request, so these have no simulator
# counterpart.
VARIABLE_DRAFT_FIELDS = (
    "num_speculative_tokens_per_batch_size",
    "enable_adaptive_verification",
)
VARIABLE_DRAFT_METHODS = ("ngram", "[ngram]", "ngram_gpu", "suffix")


# ---------------------------------------------------------------------------
# bench run
# ---------------------------------------------------------------------------

def parse_speculative_config(text: str) -> dict:
    """argparse type for ``--speculative-config``: a non-empty JSON object."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"not valid JSON: {exc}") from None
    if not isinstance(value, dict) or not value:
        raise argparse.ArgumentTypeError(
            "must be a non-empty JSON object, as vLLM's --speculative-config")
    return value


def refuse_variable_draft_count(config: dict) -> None:
    """Raise if the requested config would draft a varying number of tokens.

    vLLM accepts its config keys with hyphens or underscores, so both spellings
    are checked, and names an n-gram or suffix proposer by ``method`` or
    ``model``.
    """
    present = [
        name for name in VARIABLE_DRAFT_FIELDS
        if config.get(name) or config.get(name.replace("_", "-"))
    ]
    present += [
        f"{key}={config[key]!r}" for key in ("method", "model")
        if config.get(key) in VARIABLE_DRAFT_METHODS
    ]
    if present:
        raise ValueError(
            f"--speculative-config sets {', '.join(present)}, which varies the "
            f"draft count per step. bench records a fixed draft length only: "
            f"the simulator drafts num_speculative_tokens on every step.")


def resolved_variable_draft(speculative: Any) -> str | None:
    """Why vLLM's resolved ``SpeculativeConfig`` drafts a varying count, or None."""
    if speculative is None:
        return None
    if speculative.uses_dynamic_speculative_decoding():
        return "dynamic speculative decoding (num_speculative_tokens_per_batch_size)"
    if speculative.method in VARIABLE_DRAFT_METHODS:
        return f"method {speculative.method!r}"
    if speculative.enable_adaptive_verification:
        return "enable_adaptive_verification"
    return None


def step_counts(spec_stats: Any) -> tuple[int, int, int, list[int], list[int]]:
    """``(drafts, draft_tokens, accepted_tokens, accepted_per_pos,
    drafted_per_pos)`` for one step's ``SpecDecodingStats``, or zeros."""
    if spec_stats is None:
        return 0, 0, 0, [], []
    return (
        int(spec_stats.num_drafts),
        int(spec_stats.num_draft_tokens),
        int(spec_stats.num_accepted_tokens),
        [int(v) for v in spec_stats.num_accepted_tokens_per_pos],
        [int(v) for v in spec_stats.num_draft_tokens_per_pos],
    )


def run_summary(num_speculative_tokens: int | None, steps: list[tuple]) -> dict:
    """Sum per-step counts into ``meta.json``'s ``speculative_decoding``.

    ``steps`` holds ``step_counts`` tuples. Rates are derived the way vLLM's
    own logger derives them (``SpecDecodingLogging.log``): acceptance rate =
    accepted / drafted tokens, mean acceptance length = 1 + accepted / drafts,
    and ``position_acceptance`` = accepted per position / drafts, which is
    P(accepted >= i) and never rises with position.
    """
    width = max([num_speculative_tokens or 0]
                + [len(s[3]) for s in steps] + [len(s[4]) for s in steps])
    accepted_per_pos = [0] * width
    drafted_per_pos = [0] * width
    drafts = draft_tokens = accepted = 0
    for d, dt, a, acc_pos, dr_pos in steps:
        drafts += d
        draft_tokens += dt
        accepted += a
        for i, v in enumerate(acc_pos):
            accepted_per_pos[i] += v
        for i, v in enumerate(dr_pos):
            drafted_per_pos[i] += v
    return {
        "num_speculative_tokens": num_speculative_tokens,
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
        "num_accepted_tokens_per_pos": accepted_per_pos,
        "num_draft_tokens_per_pos": drafted_per_pos,
        "acceptance_rate": accepted / draft_tokens if draft_tokens else None,
        "mean_acceptance_length": 1 + accepted / drafts if drafts else None,
        "position_acceptance": (
            [a / drafts for a in accepted_per_pos] if drafts else None),
    }


def completion_spec_metrics(output: Any) -> dict | None:
    """The finished output's per-request acceptance payload, or None.

    vLLM 0.31 attaches ``RequestSpecDecodeMetrics`` to
    ``CompletionOutput.spec_decode_metrics`` on the finished output only, and
    only when ``per_request_spec_decode_metrics`` is not ``none``.
    """
    completions = getattr(output, "outputs", None) or []
    if not completions:
        return None
    metrics = getattr(completions[0], "spec_decode_metrics", None)
    if metrics is None or not hasattr(metrics, "to_dict"):
        return None
    return metrics.to_dict()


# ---------------------------------------------------------------------------
# bench validate
# ---------------------------------------------------------------------------

def bench_acceptance(meta: dict | None) -> dict | None:
    """Acceptance counters of a bench run, or None if it did not speculate."""
    spec = (meta or {}).get("speculative_decoding")
    if not spec:
        return None
    return {
        "n": spec.get("num_speculative_tokens"),
        "steps": spec.get("num_drafts", 0),
        "drafted": spec.get("num_draft_tokens", 0),
        "accepted": spec.get("num_accepted_tokens", 0),
        "accepted_per_pos": list(spec.get("num_accepted_tokens_per_pos") or []),
        "drafted_per_pos": list(spec.get("num_draft_tokens_per_pos") or []),
    }


# The simulator's end-of-run block (``Scheduler._print_spec_result``), one per
# instance. Counts are integers so every rate is rebuilt exactly.
_SIM_ROWS = {
    "n": re.compile(r"^Draft length \(N\):\s+(\d+)\s*$"),
    "steps": re.compile(r"^Verification steps:\s+(\d+)\s*$"),
    "drafted": re.compile(r"^Drafted tokens:\s+(\d+)\s*$"),
    "accepted": re.compile(r"^Accepted tokens:\s+(\d+)\s*$"),
}
_SIM_POS = re.compile(r"^Accepted at position (\d+):\s+(\d+) of (\d+)\s*$")


def sim_acceptance(sim_log: Path) -> dict | None:
    """Realized acceptance summed over the simulator's instances, or None.

    ``n`` is the draft length when every instance agrees and ``mixed``
    otherwise.
    """
    ns: set[int] = set()
    total = {"steps": 0, "drafted": 0, "accepted": 0}
    accepted_per_pos: list[int] = []
    drafted_per_pos: list[int] = []
    found = False
    with Path(sim_log).open() as f:
        for line in f:
            line = line.strip()
            for key, pattern in _SIM_ROWS.items():
                m = pattern.match(line)
                if m:
                    found = True
                    if key == "n":
                        ns.add(int(m.group(1)))
                    else:
                        total[key] += int(m.group(1))
                    break
            else:
                m = _SIM_POS.match(line)
                if m:
                    i = int(m.group(1)) - 1
                    while len(accepted_per_pos) <= i:
                        accepted_per_pos.append(0)
                        drafted_per_pos.append(0)
                    accepted_per_pos[i] += int(m.group(2))
                    drafted_per_pos[i] += int(m.group(3))
    if not found:
        return None
    n: int | str | None = None
    if len(ns) == 1:
        n = next(iter(ns))
    elif ns:
        n = "mixed"
    return {"n": n, **total, "accepted_per_pos": accepted_per_pos,
            "drafted_per_pos": drafted_per_pos}


def _derived(side: dict | None) -> dict:
    if side is None:
        return {}
    out = {
        "Draft length (N)": side["n"],
        "Verification steps": side["steps"],
        "Drafted tokens": side["drafted"],
        "Accepted tokens": side["accepted"],
        "Acceptance rate (%)": (side["accepted"] / side["drafted"] * 100
                                if side["drafted"] else None),
        "Mean accept length": (1 + side["accepted"] / side["steps"]
                               if side["steps"] else None),
    }
    # P(accepted >= i): accepted at position i over verification steps, the
    # rate vLLM's logger reports, on both sides.
    for i, a in enumerate(side["accepted_per_pos"], start=1):
        out[f"Position {i} accepted (%)"] = (a / side["steps"] * 100
                                              if side["steps"] else None)
    return out


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def acceptance_lines(bench: dict | None, sim: dict | None) -> tuple[list[str], list[str]]:
    """Summary rows comparing acceptance, and warnings; both empty when
    neither side speculated, so a non-speculative summary is unchanged."""
    if bench is None and sim is None:
        return [], []
    warnings = []
    if bench is None:
        warnings.append("The simulator ran speculative decoding but the vLLM run did not.")
    elif sim is None:
        warnings.append("The vLLM run used speculative decoding but the simulator did not.")
    elif bench["n"] != sim["n"]:
        warnings.append(
            f"Draft lengths differ: vLLM N={bench['n']}, simulator N={sim['n']}.")

    b, s = _derived(bench), _derived(sim)
    labels = list(b) + [k for k in s if k not in b]
    header = f"{'Speculative decoding':<25}{'vLLM':>12}{'Sim':>12}{'Diff%':>10}"
    lines = [header, "-" * len(header)]
    for label in labels:
        bv, sv = b.get(label), s.get(label)
        diff = ""
        if (label != "Draft length (N)"
                and isinstance(bv, (int, float)) and isinstance(sv, (int, float))
                and not isinstance(bv, bool) and bv):
            diff = f"{(sv - bv) / bv * 100:>+9.1f}%"
        lines.append(f"{label:<25}{_cell(bv):>12}{_cell(sv):>12}{diff:>10}".rstrip())
    lines += [f"WARNING: {w}" for w in warnings]
    lines.append("")
    return lines, warnings
