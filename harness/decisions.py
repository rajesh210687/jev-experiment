"""Small, bounded decisions. The harness prepares every option; a decider only picks.

Option 1 uses OpusDecider for everything. Option 2 uses JevDecider, which asks Jev first
and re-asks Opus only for the answers whose confidence is below the step's threshold.
"""

import json
from dataclasses import dataclass

from .config import EFFORT, JEV_MAX_QUESTIONS_PER_CALL, JEV_MAX_STATE_CHARS, JEV_MIN_CONFIDENCE
from .llm import Jev, Opus


# --- Question types (mirror Jev's Noul / Choice / Score) ---------------------------------

@dataclass
class Noul:
    """Yes/no."""
    instructions: str
    true: str
    false: str


@dataclass
class Choice:
    """Pick exactly one label from options the harness prepared (label -> description)."""
    instructions: str
    options: dict[str, str]


@dataclass
class Score:
    """Place the state on an ordered scale of 2-10 levels; levels[0] is the lowest."""
    instructions: str
    levels: list[str]


Question = Noul | Choice | Score


@dataclass
class Answer:
    value: bool | str | float  # Noul -> bool, Choice -> label, Score -> level position (0..n-1)
    confidence: float | None  # None for Opus: it gives no calibrated confidence
    by: str  # "jev" | "opus"


# --- Jev wire format -------------------------------------------------------------------------
# Request/response field names below follow OpenRouter's Decisions API docs and examples as of
# 2026-10-05. The API is alpha; re-check them against the current reference before relying on this.

def to_jev(q: Question) -> dict:
    if isinstance(q, Noul):
        return {"type": "noul", "instructions": q.instructions,
                "criteria": {"true": q.true, "false": q.false}}
    if isinstance(q, Choice):
        return {"type": "choice", "instructions": q.instructions, "criteria": q.options}
    return {"type": "score", "instructions": q.instructions, "criteria": q.levels}


def from_jev(q: Question, a: dict) -> Answer:
    if isinstance(q, Noul):
        p = float(a["noul"])  # P(true); 0.5 means "can't tell"
        return Answer(p >= 0.5, abs(2 * p - 1), "jev")
    if isinstance(q, Choice):
        return Answer(a["choice"], float(a["confidence"]), "jev")
    return Answer(float(a["score"]), float(a["confidence"]), "jev")


# --- Deciders --------------------------------------------------------------------------------

DECIDER_SYSTEM = (
    "You answer small, bounded questions for a coding agent's harness. "
    "Use only the options given. Answer every question."
)


class OpusDecider:
    def __init__(self, opus: Opus):
        self.opus = opus

    def decide(self, step: str, state: str, questions: dict[str, Question], note: str = "") -> dict[str, Answer]:
        props, lines = {}, []
        for key, q in questions.items():
            if isinstance(q, Noul):
                props[key] = {"type": "boolean"}
                lines.append(f"- {key} (true/false): {q.instructions}\n  true = {q.true}\n  false = {q.false}")
            elif isinstance(q, Choice):
                props[key] = {"type": "string", "enum": list(q.options)}
                opts = "\n".join(f"  {label}: {desc}" for label, desc in q.options.items())
                lines.append(f"- {key} (pick one label): {q.instructions}\n{opts}")
            else:
                props[key] = {"type": "integer", "enum": list(range(len(q.levels)))}
                lvls = "\n".join(f"  {i}: {desc}" for i, desc in enumerate(q.levels))
                lines.append(f"- {key} (pick a level): {q.instructions}\n{lvls}")
        schema = {"type": "object", "properties": props, "required": list(props),
                  "additionalProperties": False}
        prompt = f"<state>\n{state}\n</state>\n\nQuestions:\n" + "\n".join(lines)

        resp = self.opus.create(step, system=DECIDER_SYSTEM, effort=EFFORT["decide"],
                                messages=[{"role": "user", "content": prompt}],
                                json_schema=schema, max_tokens=4000, note=note)
        text = next(b.text for b in resp.content if b.type == "text")
        raw = json.loads(text)
        return {k: Answer(raw[k], None, "opus") for k in questions}


class JevDecider:
    def __init__(self, jev: Jev, fallback: OpusDecider):
        self.jev, self.fallback = jev, fallback

    def decide(self, step: str, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        state = state[:JEV_MAX_STATE_CHARS]
        threshold = JEV_MIN_CONFIDENCE[step]
        answers: dict[str, Answer] = {}
        unsure: dict[str, Question] = {}

        keys = list(questions)
        for i in range(0, len(keys), JEV_MAX_QUESTIONS_PER_CALL):
            batch = {k: questions[k] for k in keys[i:i + JEV_MAX_QUESTIONS_PER_CALL]}
            try:
                raw, rec = self.jev.decide(step, state, {k: to_jev(q) for k, q in batch.items()})
            except Exception as e:  # network error, 4xx/5xx, schema drift: Opus decides the batch
                print(f"  [jev] {step}: {type(e).__name__}: {e} -> falling back to Opus")
                unsure.update(batch)
                continue
            for k, q in batch.items():
                a = from_jev(q, raw[k])
                if a.confidence >= threshold:
                    answers[k] = a
                else:
                    unsure[k] = q
                    rec.low_confidence += 1

        # The confidence gate: anything Jev was unsure about goes to Opus, in one call.
        if unsure:
            answers.update(self.fallback.decide(step, state, unsure, note="jev-fallback"))
        return answers
