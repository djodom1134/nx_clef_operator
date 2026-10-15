from clef_operator.motion import (
    GRID_H, GRID_W, MotionRank, WindowMotion, bounding_box, clusters, empty_grid, format_mask, parse_mask,
)

# A real mask from the Front porch camera.
PORCH = (
    "0,0,0,44,7;0,0,7,29,2;0,0,9,27,1;0,34,7,10,6;0,0,10,26,4;0,34,13,1,1;0,0,14,9,1;0,15,14,11,1;0,22,15,4,1;"
    "0,0,15,4,2;0,0,17,5,2;0,43,19,1,2;0,0,19,3,4;0,0,23,5,1;0,0,24,7,4;0,42,21,2,7;0,0,28,14,2;0,41,28,3,2;"
    "0,0,30,26,1;0,0,31,28,1;0,37,30,7,2;5,29,7,5,2;5,27,9,7,1;5,35,13,9,1;5,9,14,6,1;5,4,15,18,1;5,26,10,8,6;"
    "5,4,16,30,1;5,36,14,8,3;5,5,17,23,2;5,37,17,7,2;5,37,19,6,2;5,3,19,25,4;5,5,23,23,1;5,7,24,19,4;5,37,21,5,7;"
    "5,14,28,12,2;5,37,28,4,2;7,34,14,2,3;7,28,17,9,7;7,26,24,11,7;7,28,31,9,1"
)


def test_mask_round_trip_on_real_mask():
    grid = parse_mask(PORCH)
    assert grid[0][0] == 0 and grid[17][30] == 7
    assert parse_mask(format_mask(grid)) == grid


def test_format_merges_rows():
    grid = empty_grid(5)
    for y in range(4, 8):
        for x in range(10, 20):
            grid[y][x] = 0
    text = format_mask(grid)
    assert "0,10,4,10,4" in text
    assert parse_mask(text) == grid


def test_rank_still_active_onset():
    motion = WindowMotion()
    for i in range(60):
        motion.add_frame({(3, 3)} if i < 10 else set())
    assert motion.rank(active_share=0.05) == MotionRank.ACTIVE
    motion.onset = True
    assert motion.rank() == MotionRank.ONSET
    assert WindowMotion(frames=60).rank() == MotionRank.STILL


def test_rank_ignores_nuisance_cells_and_respects_zone():
    motion = WindowMotion()
    for _ in range(20):
        motion.add_frame({(0, 0)})  # a flag in the corner
    nuisance = empty_grid()
    nuisance[0][0] = 1
    assert motion.rank(ignore=nuisance) == MotionRank.STILL
    zone = empty_grid()
    zone[GRID_H - 1][GRID_W - 1] = 1
    assert motion.rank(only=zone) == MotionRank.STILL
    assert motion.rank() == MotionRank.ACTIVE


def test_clusters_and_bounding_box():
    cells = {(0, 0), (1, 1), (10, 10), (11, 10)}
    groups = sorted(clusters(cells), key=len)
    assert [len(g) for g in groups] == [2, 2]
    x, y, w, h = bounding_box({(10, 10), (11, 10)})
    assert (round(x * GRID_W), round(y * GRID_H), round(w * GRID_W), round(h * GRID_H)) == (10, 10, 2, 1)


def test_fast_crossing_of_a_zone_counts_as_motion():
    road = empty_grid()
    for x in range(10):
        road[5][x] = 1
    motion = WindowMotion()
    for f in range(210):  # 30 s at 7 fps; a car crosses the 10-cell road in 14 frames
        motion.add_frame({(f * 10 // 14, 5)} if f < 14 else set())
    assert motion.rank(active_share=0.05, only=road) == MotionRank.ACTIVE


def test_continuous_motion_counts_but_scattered_noise_does_not():
    car, noise = WindowMotion(), WindowMotion()
    for f in range(213):
        car.add_frame({(20 + f - 100, 3)} if 100 <= f < 110 else set())  # 10 frames in a row: under 5%
        noise.add_frame({(f % 44, 7)} if f % 25 == 0 else set())  # 9 single frames (4%), scattered
    assert car.rank(active_share=0.05, fps=7) == MotionRank.ACTIVE
    assert noise.rank(active_share=0.05, fps=7) == MotionRank.STILL
