from clef_operator.events import EventState, Question


def test_hallway_fight_opens_once_and_closes():
    state = EventState(Question("fight", "Is anyone fighting?"))
    windows = [(0, 30, 0.02), (20, 50, 0.1), (40, 70, 0.9), (60, 90, 0.7), (80, 110, 0.2)]
    transitions = [state.update(*w) for w in windows]
    opened = [t for t in transitions if t and t.kind == "open"]
    closed = [t for t in transitions if t and t.kind == "close"]
    assert len(opened) == 1 and opened[0].time == 40 and opened[0].probability == 0.9
    assert len(closed) == 1 and closed[0].time == 90 and closed[0].probability == 0.9


def test_close_after_two_quiet_windows():
    state = EventState(Question("q", "?", close_after=2))
    assert state.update(0, 30, 0.95).kind == "open"
    assert state.update(20, 50, 0.1) is None
    assert state.update(40, 70, None).kind == "close"  # a skipped window counts as below


def test_choice_and_score_alert_probability():
    choice = Question("c", "What is moving?", "choice", ["flag", "person", "tree"], alert_on=["person"])
    assert choice.alert_probability({"flag": 0.7, "person": 0.2, "tree": 0.1}) == 0.2
    crowd = Question("s", "How crowded?", "score", ["empty", "light", "busy", "packed"], alert_on=["busy"])
    assert round(crowd.alert_probability({"empty": 0.1, "light": 0.2, "busy": 0.3, "packed": 0.4}), 6) == 0.7
