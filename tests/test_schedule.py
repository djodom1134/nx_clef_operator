from clef_operator.motion import MotionRank
from clef_operator.schedule import Job, Priority, Scheduler


def job(camera="c1", group="g", priority=Priority.NORMAL, end=100.0, motion=MotionRank.ACTIVE, fps=2.0):
    return Job(camera, group, priority, end, motion, fps)


def test_class_then_motion_then_deadline():
    s = Scheduler()
    s.add(job("a", priority=Priority.NORMAL, motion=MotionRank.ONSET), now=100)
    s.add(job("b", priority=Priority.CRITICAL, motion=MotionRank.STILL), now=100)
    s.add(job("c", priority=Priority.CRITICAL, motion=MotionRank.ONSET), now=100)
    assert [s.next(101).camera_id for _ in range(3)] == ["c", "b", "a"]
    assert s.next(101) is None


def test_merge_keeps_newest_window_for_routine_classes():
    s = Scheduler()
    s.add(job(end=100), now=100)
    s.add(job(end=120), now=120)
    assert len(s) == 1 and s.next(121).window_end == 120
    assert s.skipped["NORMAL:merged"] == 1
    assert [j.window_end for j in s.take_dropped()] == [100] and s.take_dropped() == []


def test_critical_windows_are_never_merged_or_dropped():
    s = Scheduler()
    s.add(job(priority=Priority.CRITICAL, end=100), now=100)
    s.add(job(priority=Priority.CRITICAL, end=120), now=120)
    assert len(s) == 2
    assert s.next(10_000).window_end == 100


def test_expired_jobs_are_dropped_and_counted():
    s = Scheduler()
    s.add(job(priority=Priority.HIGH, end=100), now=100)
    assert s.next(161) is None
    assert s.skipped["HIGH:expired"] == 1


def test_low_classes_capped_at_2_fps():
    s = Scheduler()
    s.add(job(priority=Priority.LOW, fps=8), now=0)
    assert s.next(1).fps == 2


def test_maintenance_promoted_after_45_minutes():
    s = Scheduler()
    s.add(job("old", priority=Priority.MAINTENANCE, end=0), now=0)
    s.add(job("new", priority=Priority.NORMAL, end=2_790), now=2_790)
    first = s.next(2_801)
    assert first.camera_id == "old" and first.priority == Priority.NORMAL


def test_a_waiting_low_job_runs_before_its_target_passes_despite_fresh_high_ones():
    s = Scheduler()
    s.add(job("low", priority=Priority.LOW, end=100), now=100)  # due at 120
    s.add(job("high", priority=Priority.HIGH, end=110), now=110)  # due at 122
    assert s.next(111).camera_id == "low"


def test_jobs_arriving_together_still_run_in_class_order():
    s = Scheduler()
    for name, priority in (("low", Priority.LOW), ("normal", Priority.NORMAL), ("crit", Priority.CRITICAL)):
        s.add(job(name, group=name, priority=priority), now=100)
    assert [s.next(100).camera_id for _ in range(3)] == ["crit", "normal", "low"]


def test_frame_rates_are_lowered_lowest_priority_first_to_stay_on_time():
    s = Scheduler()
    # A 4 fps window takes about 4.6 s. One Critical, then High, Normal and Low ones, all at once: the High one would
    # finish at 9.3 s (target 12: fine), the Normal one at 14 s (target 16: fine), the Low one at 16.2 s (fine).
    # Add a second High one and the queue no longer fits: only jobs ahead of a late one can help it.
    jobs = [job(f"c{i}", group=f"g{i}", priority=p, fps=4.0)
            for i, p in enumerate([Priority.CRITICAL, Priority.HIGH, Priority.HIGH, Priority.NORMAL, Priority.LOW])]
    for j in jobs:
        s.add(j, now=100)
    s.next(100)
    crit, high1, high2, normal = jobs[0], jobs[1], jobs[2], jobs[3]
    assert crit.fps == 4.0  # only ever lowered for another Critical job
    assert min(high1.fps, high2.fps) == 2.0 and normal.fps <= 4.0
    assert "CRITICAL" not in s.lowered and s.lowered["HIGH"] >= 1


def test_critical_jobs_give_up_frames_only_for_each_other_and_never_below_2_fps():
    s = Scheduler()
    jobs = [job(f"c{i}", group=f"g{i}", priority=Priority.CRITICAL, fps=4.0) for i in range(3)]
    for j in jobs:
        s.add(j, now=100)
    s.next(100)
    assert all(j.fps >= 2.0 for j in jobs) and s.lowered["CRITICAL"] >= 1
    s2 = Scheduler()
    s2.add(job("crit", group="a", priority=Priority.CRITICAL, fps=4.0), now=100)
    for i in range(4):
        s2.add(job(f"n{i}", group=f"n{i}", priority=Priority.NORMAL, fps=4.0), now=100)
    s2.next(100)
    assert "CRITICAL" not in s2.lowered  # Normal jobs running late never cost a Critical one its frames


def test_an_idle_queue_keeps_the_requested_frame_rate():
    s = Scheduler()
    s.add(job(priority=Priority.LOW, fps=2.0), now=100)
    s.add(job("b", group="b", priority=Priority.NORMAL, fps=4.0), now=100)
    assert s.next(100).fps == 4.0 and s.next(100).fps == 2.0 and not s.lowered


def test_the_cost_model_learns_from_answers():
    s = Scheduler()
    before = s.cost.per_token_s
    for _ in range(30):
        s.cost.learn(9000, 0.25 + 9000 * 0.0007)  # Clef got slower
    assert abs(s.cost.per_token_s - 0.0007) < 0.00005 > 0 and s.cost.per_token_s > before
