from datetime import datetime

from clef_operator.profile import ProfileStore, slot_key


def at(day: int, hour: int) -> float:
    return datetime(2026, 10, day, hour, 30).timestamp()  # Oct 2026: the 5th..9th and 12th..16th are weekdays


def survey(people_none: float, moving: float = 0.0, animal: float = 0.0) -> dict:
    rest = (1 - people_none) / 3
    return {
        "_people": {"none": people_none, "one": rest, "a few": rest, "many": rest},
        "_vehicles": {"none": 1 - moving, "parked": 0.0, "moving": moving},
        "_animal": {"true": animal, "false": 1 - animal},
    }


def test_no_profile_until_five_days(tmp_path):
    store = ProfileStore(tmp_path / "p.json")
    for day in (12, 13, 14, 15):
        store.observe("porch", at(day, 12), survey(0.1, moving=0.7))
    assert store.text("porch", at(16, 12)) == ""
    store.observe("porch", at(16, 12), survey(0.1, moving=0.7))
    text = store.text("porch", at(16, 12))
    assert "people are usually present" in text and "Vehicles usually move through" in text


def test_slots_separate_hours_and_persist(tmp_path):
    path = tmp_path / "p.json"
    store = ProfileStore(path)
    for day in (5, 6, 7, 8, 9):
        store.observe("basement", at(day, 3), survey(0.99))
    reloaded = ProfileStore(path)
    assert "people are rarely present" in reloaded.text("basement", at(12, 3))
    assert reloaded.text("basement", at(12, 15)) == ""
    assert slot_key(at(10, 3)) == "weekend-03"


def test_unusual_person_at_night_and_survey_cadence():
    store = ProfileStore()
    for day in (5, 6, 7, 8, 9):
        store.observe("basement", at(day, 3), survey(0.99))
    assert store.unusual("basement", at(12, 3), 0.95)
    assert not store.unusual("basement", at(12, 3), 0.3)
    assert store.survey_due("basement", 1000.0)
    assert not store.survey_due("basement", 1500.0)
    assert store.survey_due("basement", 1000.0 + 3601)
