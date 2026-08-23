"""Contract-level tests for the camera module (no hardware required)."""

import pytest

from camera.interfaces.contracts import ListenerSet, notify_listeners
from camera.interfaces.types import ConnectionState, FrameStamp


class TestConnectionStateContract:
    def test_values_match_frontend_contract_exactly(self):
        # frontend/services/core/types.ts — ConnectionState union
        expected = {"disconnected", "connecting", "connected", "error"}
        assert {state.value for state in ConnectionState} == expected

    def test_is_string_enum_for_cheap_json_serialization(self):
        assert isinstance(ConnectionState.CONNECTED, str)
        assert ConnectionState.CONNECTED == "connected"


class TestFrameStamp:
    def test_epoch_ms_derives_from_wallclock(self):
        wallclock_ns = 1_700_000_000_123_456_789
        stamp = FrameStamp(
            sequence=1,
            monotonic_ns=987_654_321,
            wallclock_ns=wallclock_ns,
            fps=30.0,
        )
        assert stamp.epoch_ms == 1_700_000_000_123

    def test_frozen_value_object(self):
        stamp = FrameStamp(1, 0, 0, 0.0)
        with pytest.raises(Exception):
            stamp.sequence = 2  # type: ignore[misc]


class TestListenerSet:
    def test_unsubscribe_is_idempotent(self):
        registry: ListenerSet[object] = ListenerSet()
        received = []
        unsubscribe = registry.add(received.append)

        unsubscribe()
        unsubscribe()  # must not raise

        notify_listeners(registry.snapshot(), "event")
        assert received == []

    def test_snapshot_isolates_from_mutation_during_notification(self):
        registry = ListenerSet()
        seen_first = []
        unsubscribe = registry.add(seen_first.append)

        snapshot = registry.snapshot()
        unsubscribe()

        assert len(snapshot) == 1
        assert registry.snapshot() == ()

    def test_notify_isolates_listener_exceptions(self):
        registry = ListenerSet()
        good_received = []

        def bad_listener(_):
            raise RuntimeError("listener bug")

        registry.add(bad_listener)
        unsubscribe_good = registry.add(good_received.append)

        try:
            notify_listeners(registry.snapshot(), 42)
        except Exception as exc:  # notification must never propagate
            pytest.fail(f"notify_listeners raised: {exc}")

        assert good_received == [42]
        unsubscribe_good()
