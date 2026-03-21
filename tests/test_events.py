"""Tests for the event bus (agent/events.py)."""

import pytest

from agent.events import (
    AGENT_COMPLETE,
    AGENT_ERROR,
    AGENT_PROGRESS,
    AGENT_STEP,
    TOOL_EXECUTE,
    TOOL_RESULT,
    EventBus,
    event_bus,
)


class TestEventBusSubscribeEmit:
    """Tests for basic subscribe / emit flow."""

    def test_sync_handler_receives_event(self):
        bus = EventBus()
        received = []
        bus.subscribe("test.event", lambda **kw: received.append(kw))
        bus.emit("test.event", key="value")
        assert received == [{"key": "value"}]

    def test_multiple_handlers(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe("ev", lambda **kw: a.append(kw))
        bus.subscribe("ev", lambda **kw: b.append(kw))
        bus.emit("ev", x=1)
        assert len(a) == 1 and len(b) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        unsub = bus.subscribe("ev", lambda **kw: received.append(1))
        bus.emit("ev")
        unsub()
        bus.emit("ev")
        assert received == [1]

    def test_emit_unknown_event_is_noop(self):
        bus = EventBus()
        bus.emit("nonexistent")  # should not raise

    def test_handler_error_does_not_crash(self):
        bus = EventBus()
        bus.subscribe("ev", lambda **kw: 1 / 0)  # ZeroDivisionError
        bus.emit("ev")  # should not raise

    def test_clear_specific_event(self):
        bus = EventBus()
        received = []
        bus.subscribe("a", lambda **kw: received.append("a"))
        bus.subscribe("b", lambda **kw: received.append("b"))
        bus.clear("a")
        bus.emit("a")
        bus.emit("b")
        assert received == ["b"]

    def test_clear_all(self):
        bus = EventBus()
        received = []
        bus.subscribe("a", lambda **kw: received.append("a"))
        bus.subscribe("b", lambda **kw: received.append("b"))
        bus.clear()
        bus.emit("a")
        bus.emit("b")
        assert received == []


class TestEventBusAsync:
    """Tests for async handler support."""

    @pytest.mark.asyncio
    async def test_emit_async_awaits_handlers(self):
        bus = EventBus()
        received = []

        async def handler(**kw):
            received.append(kw)

        bus.subscribe("ev", handler)
        await bus.emit_async("ev", val=42)
        assert received == [{"val": 42}]

    @pytest.mark.asyncio
    async def test_emit_async_handles_mixed_handlers(self):
        bus = EventBus()
        sync_calls = []
        async_calls = []

        def sync_handler(**kw):
            sync_calls.append(kw)

        async def async_handler(**kw):
            async_calls.append(kw)

        bus.subscribe("ev", sync_handler)
        bus.subscribe("ev", async_handler)
        await bus.emit_async("ev", x=1)
        assert len(sync_calls) == 1
        assert len(async_calls) == 1

    @pytest.mark.asyncio
    async def test_emit_async_handler_error_does_not_crash(self):
        bus = EventBus()

        async def bad_handler(**kw):
            raise RuntimeError("boom")

        bus.subscribe("ev", bad_handler)
        await bus.emit_async("ev")  # should not raise


class TestEventBusHasSubscribers:
    def test_has_subscribers_true(self):
        bus = EventBus()
        bus.subscribe("ev", lambda **kw: None)
        assert bus.has_subscribers("ev") is True

    def test_has_subscribers_false(self):
        bus = EventBus()
        assert bus.has_subscribers("ev") is False


class TestEventConstants:
    """Verify all expected constants are defined."""

    def test_event_constants_are_strings(self):
        for const in [
            AGENT_PROGRESS,
            AGENT_COMPLETE,
            AGENT_ERROR,
            AGENT_STEP,
            TOOL_EXECUTE,
            TOOL_RESULT,
        ]:
            assert isinstance(const, str) and "." in const


class TestGlobalEventBus:
    """The module-level event_bus singleton should work."""

    def test_global_bus_is_event_bus(self):
        assert isinstance(event_bus, EventBus)

    def test_global_bus_subscribe_emit(self):
        received = []
        unsub = event_bus.subscribe("_test_global", lambda **kw: received.append(1))
        event_bus.emit("_test_global")
        unsub()
        assert received == [1]
