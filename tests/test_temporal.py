from src.temporal import TemporalState


def test_temporal_state_waits_until_duration_is_met():
    state = TemporalState(required_seconds=2.0)

    assert state.update(True, now=10.0) is False
    assert state.update(True, now=11.5) is False
    assert state.update(True, now=12.0) is True


def test_temporal_state_resets_when_condition_clears():
    state = TemporalState(required_seconds=2.0)

    assert state.update(True, now=10.0) is False
    assert state.update(False, now=11.0) is False
    assert state.update(True, now=20.0) is False
    assert state.update(True, now=21.0) is False
