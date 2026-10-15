"""Questions and the answer-to-event state machine (PRD: Pipeline, "Answer to event").

One ``EventState`` per (camera, question). An event opens when the alert probability reaches the on-threshold,
stays open while windows stay at or above the off-threshold, and closes after ``close_after`` windows in a row below
it. A skipped window (no motion, for a question that needs motion) counts as below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import clef


@dataclass
class Question:
    id: str
    text: str
    kind: str = "noul"  # noul | choice | score
    options: list[str] = field(default_factory=list)  # choice / score options, in order for score
    alert_on: list[str] = field(default_factory=list)  # choice: options that alert; score: [lowest alerting level]
    on_threshold: float = 0.8
    off_threshold: float = 0.5
    close_after: int = 1
    zone: str | None = None

    def to_clef(self) -> dict:
        if self.kind == "noul":
            return clef.noul(self.text)
        if self.kind == "choice":
            return clef.choice(self.text, {option: option for option in self.options})
        return clef.score(self.text, list(self.options))

    def alert_probability(self, answer: dict[str, float]) -> float:
        if self.kind == "noul":
            return answer["true"]
        if self.kind == "choice":
            return sum(answer.get(option, 0.0) for option in self.alert_on)
        # score: probability at or above the lowest alerting level. Answers may be keyed by level text or index.
        lowest = self.options.index(self.alert_on[0]) if self.alert_on else len(self.options) - 1
        total = 0.0
        for index, level in enumerate(self.options):
            if index >= lowest:
                total += answer.get(level, answer.get(str(index), 0.0))
        return total


@dataclass
class Transition:
    kind: str  # "open" | "close"
    time: float  # epoch seconds: window start for open, last positive window end for close
    probability: float


@dataclass
class EventState:
    question: Question
    active: bool = False
    started: float = 0.0
    last_positive_end: float = 0.0
    peak: float = 0.0
    below: int = 0

    def update(self, window_start: float, window_end: float, probability: float | None) -> Transition | None:
        """Feed one window's alert probability (None = skipped). Returns a transition, if one happened."""
        p = 0.0 if probability is None else probability
        q = self.question
        if not self.active:
            if p >= q.on_threshold:
                self.active, self.started, self.last_positive_end, self.peak, self.below = True, window_start, window_end, p, 0
                return Transition("open", window_start, p)
            return None
        if p >= q.off_threshold:
            self.below, self.last_positive_end, self.peak = 0, window_end, max(self.peak, p)
            return None
        self.below += 1
        if self.below >= q.close_after:
            self.active = False
            return Transition("close", self.last_positive_end, self.peak)
        return None
