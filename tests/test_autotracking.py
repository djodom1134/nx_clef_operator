from clef_operator.gateway import AutoTracking


class FakeNx:
    def __init__(self):
        self.values = {"AutoTracking.Enable": True, "MotionDetection.Enable": True}
        self.writes = []

    def _request(self, method, path, body=None):
        return {"userEnabledAnalyticsEngineIds": ["clef"], "compatibleAnalyticsEngineIds": ["milesight"]}  # as Nx 6.1

    def device_agent_settings(self, engine, camera):
        if engine != "milesight":
            raise RuntimeError("no settings")
        return {"values": dict(self.values)}

    def set_device_agent_settings(self, engine, camera, updates):
        self.writes.append((engine, updates))
        self.values.update(updates)


def test_the_switch_is_found_written_only_on_change_and_restored():
    nx = FakeNx()
    tracking = AutoTracking(nx, "cam")
    assert (tracking.engine, tracking.key, tracking.original) == ("milesight", "AutoTracking.Enable", True)
    tracking.set(False)
    tracking.set(False)
    assert nx.writes == [("milesight", {"AutoTracking.Enable": False})]
    tracking.restore()
    assert nx.values["AutoTracking.Enable"] is True and len(nx.writes) == 2


def test_the_operators_setting_survives_a_crash(tmp_path):
    saved = tmp_path / "autotracking.json"
    nx = FakeNx()
    AutoTracking(nx, "cam", saved).set(False)  # hunting: off; then the gateway dies without restoring
    again = AutoTracking(nx, "cam", saved)  # the camera now reads off
    assert again.original is True  # but the operator had it on
    again.restore()
    assert nx.values["AutoTracking.Enable"] is True
