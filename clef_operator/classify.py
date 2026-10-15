"""Question classifier (PRD: Adaptive frame rate; Priority and scheduling).

One text-only Clef request per saved question picks its frame rate, priority class and whether it needs motion.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clef import Clef, choice, noul, score
from .schedule import Priority

FPS_OPTIONS = {
    "1": "The thing asked about stays the same for minutes at a time, like a parked car or a dirty lens.",
    "2": "It changes over tens of seconds, like people walking through or a crowd building up.",
    "4": "It changes within a few seconds, like someone falling down or a door opening.",
    "8": "It is brief, fast motion, like a punch, a shove or a thrown object.",
    "10": "It is split-second action that could be missed between frames.",
}
PRIORITY_LEVELS = ["maintenance", "low", "normal", "high", "critical"]  # ordered low -> high for the score type
PRIORITY_BY_LEVEL = {
    "maintenance": Priority.MAINTENANCE, "low": Priority.LOW, "normal": Priority.NORMAL,
    "high": Priority.HIGH, "critical": Priority.CRITICAL,
}
CLOSE = 0.1  # top two options this close: take the more cautious one (higher fps, higher priority)


@dataclass
class Classification:
    fps: float
    priority: Priority
    needs_motion: bool
    fps_probabilities: dict[str, float]
    priority_probabilities: dict[str, float]
    needs_motion_probability: float

    def summary(self) -> str:
        return (f"{self.fps:g} fps ({self.fps_probabilities[str(int(self.fps))]:.2f}) · "
                f"{self.priority.name.title()} ({self.priority_probabilities[self.priority.name.lower()]:.2f}) · "
                f"{'needs' if self.needs_motion else 'no'} motion ({self.needs_motion_probability:.2f})")


def _by_level(probabilities: dict[str, float], levels: list[str]) -> dict[str, float]:
    """Score answers may come back keyed by level text or by index; normalize to level text."""
    return {level: probabilities.get(level, probabilities.get(str(i), 0.0)) for i, level in enumerate(levels)}


def _cautious(probabilities: dict[str, float], order: list[str]) -> str:
    """Highest-probability option, but when the runner-up is within CLOSE and later in ``order``, take it."""
    ranked = sorted(order, key=lambda o: probabilities[o], reverse=True)
    best, second = ranked[0], ranked[1]
    if probabilities[best] - probabilities[second] <= CLOSE and order.index(second) > order.index(best):
        return second
    return best


def classify(clef: Clef, question: str, camera: str = "", description: str = "", interests: str = "") -> Classification:
    state = f"A plain-language question a security operator wants answered continuously about a camera's video: \"{question}\"."
    if camera:
        state += f" Camera: {camera}."
    if description:
        state += f" About this camera: {description}"
    if interests:
        state += f" What matters at this site: {interests}"
    result = clef.ask(state, {
        "fps": choice("How fast does the thing this question asks about change, and how briefly might it appear?", FPS_OPTIONS),
        "priority": score(
            "How urgently would a security operator need to act if the answer were yes? "
            "Consider harm to people first, then property, then equipment upkeep.",
            [f"{level}" for level in PRIORITY_LEVELS],
        ),
        "needs_motion": noul("Can the answer only be yes while something in view is moving?"),
    })
    fps_p = result.answers["fps"]
    priority_p = _by_level(result.answers["priority"], PRIORITY_LEVELS)
    fps = _cautious(fps_p, list(FPS_OPTIONS))
    level = _cautious(priority_p, PRIORITY_LEVELS)
    motion_p = result.answers["needs_motion"]["true"]
    return Classification(float(fps), PRIORITY_BY_LEVEL[level], motion_p >= 0.5, fps_p, priority_p, motion_p)
