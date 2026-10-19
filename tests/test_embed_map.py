from clef_operator.embed_map import EmbedMap, pack, tile_view, unpack
from clef_operator.hunter import Hunter, HunterConfig, View, ViewStats, tile_boxes


def unit(*xs):
    n = sum(x * x for x in xs) ** 0.5
    return [x / n for x in xs]


def test_a_look_adds_its_frame_and_tiles_each_at_its_own_direction(tmp_path):
    m = EmbedMap(tmp_path / "map.json")
    view = View(190.0, -10.0, 36.9)
    boxes = tile_boxes(3)
    m.add_look(view, unit(1, 0, 0), [unit(0, 1, 0)] * 9, boxes, b"jpeg", now=1000.0)
    assert len(m.entries) == 10
    middle = tile_view(view, boxes[4])
    assert abs(middle.pan - 190.0) < 0.5 and abs(middle.fov - 36.9 / 2) < 0.5  # the centre tile: same aim, half as wide
    corner = tile_view(view, boxes[0])
    assert corner.pan < 190.0 and corner.tilt > -10.0  # top left: further left and higher


def test_a_later_look_at_the_same_spot_replaces_the_earlier_one(tmp_path):
    m = EmbedMap(tmp_path / "map.json")
    view = View(190.0, -10.0, 36.9)
    m.add_look(view, unit(1, 0, 0), [], [], b"a", now=1000.0)
    m.add_look(View(191.0, -10.5, 36.9), unit(0, 1, 0), [], [], b"b", now=2000.0)
    assert len(m.entries) == 1 and m.entries[0]["t"] == 2000.0


def test_search_ranks_by_similarity_one_result_per_spot(tmp_path):
    m = EmbedMap(tmp_path / "map.json")
    m.add_look(View(100.0, -10.0, 36.9), unit(1, 0, 0), [], [], b"wall", now=1.0)
    m.add_look(View(190.0, -10.0, 36.9), unit(0, 1, 0), [], [], b"picnic", now=2.0)
    m.add_look(View(250.0, -30.0, 36.9), unit(0.3, 1, 0), [], [], b"lawn", now=3.0)
    found = m.search([unit(0, 1, 0)], top=2)
    assert [r["pan"] for r in found["top"]] == [190.0, 250.0] and len(found["map"]) == 3
    assert m.thumb(found["top"][0]["shot"]) == b"picnic"
    contrast = m.search([unit(0, 1, 0)], top=1, negatives=[unit(0.3, 1, 0)])  # like a picnic, not like the lawn
    assert contrast["top"][0]["pan"] == 190.0 and contrast["top"][0]["score"] > 0


def test_the_map_survives_a_restart_and_drops_unused_thumbnails(tmp_path):
    m = EmbedMap(tmp_path / "map.json")
    m.add_look(View(190.0, -10.0, 36.9), unit(1, 2, 3), [], [], b"old", now=1000.0)
    old_shot = m.entries[0]["shot"]
    m.add_look(View(190.0, -10.0, 36.9), unit(3, 2, 1), [], [], b"new", now=2000.0)
    m.save()
    again = EmbedMap(tmp_path / "map.json")
    assert len(again.entries) == 1 and again.thumb(old_shot) is None and again.thumb(again.entries[0]["shot"]) == b"new"
    assert all(abs(a - b) < 1e-3 for a, b in zip(again.entries[0]["vec"], unit(3, 2, 1)))
    assert unpack(pack([0.5, -0.25])) == [0.5, -0.25]


def test_look_there_goes_next_and_then_the_hunt_carries_on():
    from tests.test_hunter import FakePtz
    ptz = FakePtz(View(184.0, -12.0, 36.9))
    looked = []

    def ask(camera_id, jpeg, questions):
        looked.append(ptz.position())
        hunter.stop(wait=False)
        return {q: {"true": 0.02} for q in questions}

    hunter = Hunter(HunterConfig("cam", pan_min=184, pan_max=184.5, tilts=[-12.0], rest_s=0, settle_s=0, tiles=1),
                    ptz, ask, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.stats[View(184.0, -12.0, 36.9)] = ViewStats(visits=5, scored=True, scene=0.5)  # surveyed already
    hunter.home = View(184.0, -12.0, 36.9)
    hunter.look_at(View(230.0, -20.0, 36.9))
    hunter.run()
    assert looked and abs(looked[0].pan - 230.0) <= 2.5 and abs(looked[0].tilt + 20.0) <= 2.5
