from clef_operator.action import MIN_SEEN, ActionWatch, ViewMotion, box, decode, encode

TREE = {(c, r) for c in range(30, 34) for r in range(2, 6)}  # top right: a tree that waves on and off
PERSON = {(10, 20), (10, 21), (11, 20), (11, 21)}  # lower left: someone walking in


def watched(frames: int = 700) -> ActionWatch:
    """A view watched long enough to have learned its tree: waving in half the frames."""
    watch = ActionWatch(ViewMotion())
    for i in range(frames):
        watch.add(i * 0.5, TREE if i % 2 else set())
    return watch


def test_the_grid_round_trips_through_the_plugin_encoding():
    cells = {(0, 0), (43, 31), (10, 20)}
    assert decode(encode(cells)) == cells


def test_waving_trees_are_learned_as_boring_and_never_action():
    watch = watched()
    assert all(watch.motion.boring(cell) for cell in TREE)
    t = 700 * 0.5
    assert all(watch.add(t + i * 0.5, TREE) is None for i in range(6))  # the tree keeps waving: nothing


def test_something_new_that_keeps_moving_for_a_second_is_action():
    watch = watched()
    t = 700 * 0.5
    for i in range(8):  # quiet, apart from the tree, for four seconds
        watch.add(t + i * 0.5, TREE if i % 2 else set())
    t += 4.0
    assert watch.add(t, PERSON | TREE) is None  # one frame could be noise
    found = watch.add(t + 0.5, {(c + 1, r) for c, r in PERSON})  # still there, a step on: action
    assert found is not None
    x, y, w, h = found
    assert x < 0.4 and y > 0.4  # lower left, where the person is


def test_one_noisy_frame_is_not_action():
    watch = watched()
    t = 700 * 0.5 + 4.0
    assert watch.add(t, PERSON) is None
    assert watch.add(t + 0.5, set()) is None


def test_nothing_counts_before_a_view_has_been_watched_a_while():
    watch = ActionWatch(ViewMotion())
    for i in range(MIN_SEEN - 5):
        assert watch.add(i * 0.5, PERSON if i > 10 else set()) is None


def test_the_learned_view_survives_a_restart():
    watch = watched()
    again = ViewMotion.from_json(watch.motion.to_json())
    assert again.seen == watch.motion.seen and again.boring((31, 3)) and not again.boring((10, 20))


def test_the_box_grows_round_the_cluster_and_stays_in_the_frame():
    x, y, w, h = box({(43, 31)})
    assert 0.0 <= x and x + w <= 1.0 and 0.0 <= y and y + h <= 1.0 and w > 0 and h > 0


def test_leaves_learned_from_one_view_are_a_nuisance_from_an_overlapping_one():
    from clef_operator.action import NuisanceMap
    from clef_operator.hunter import View
    nuisance = NuisanceMap()
    here, next_door = View(190.0, -10.0, 37.0), View(200.0, -10.0, 37.0)  # 10 degrees apart: they overlap
    leaves = {(c, r) for c in range(36, 40) for r in range(14, 18)}  # right of centre at 190: about pan 202
    for i in range(200):
        nuisance.observe(here, leaves if i % 3 == 0 else set())  # moving a third of the time
    busy = nuisance.busy(next_door)
    assert busy and all(abs(c - 25) <= 4 for c, _ in busy)  # the same leaves, just right of centre from 200
    assert not nuisance.busy(View(250.0, -10.0, 37.0))  # nowhere else


def test_the_nuisance_map_and_its_fruitless_chases_survive_a_restart():
    from clef_operator.action import NuisanceMap
    from clef_operator.hunter import View
    nuisance = NuisanceMap()
    for i in range(100):
        nuisance.observe(View(190.0, -10.0, 37.0), {(10, 10)} if i % 2 else set())
    nuisance.chased(1000.0, 200.0, -22.0)
    nuisance.chased(1100.0, 201.0, -23.0)
    again = NuisanceMap.from_json(nuisance.to_json())
    assert again.cells == nuisance.cells and again.busy(View(190.0, -10.0, 37.0)) == nuisance.busy(View(190.0, -10.0, 37.0))
    assert again.fruitless(1200.0, 200.5, -22.5, found_since=0.0) > 1.9
    assert again.fruitless(1200.0, 200.5, -22.5, found_since=1050.0) < 1.1  # someone found there in between
    assert again.fruitless(1200.0 + 2 * 86400, 200.5, -22.5, found_since=0.0) < 0.6  # fades over days
