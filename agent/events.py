"""Event bus for decoupled component communication.

Provides a publish/subscribe mechanism so components can communicate
without direct function call dependencies. Supports both sync and
async event handlers.

Usage:
    from agent.events import event_bus, AGENT_PROGRESS

    # Subscribe
    def on_progress(**data):
        print(f"Step {data['iteration']}")

    unsub = event_bus.subscribe(AGENT_PROGRESS, on_progress)

    # Emit
    await event_bus.emit_async(AGENT_PROGRESS, iteration=1, status="thinking")

    # Unsubscribe
    unsub()
"""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

AGENT_PROGRESS = "agent.progress"
AGENT_CONFIRM_REQUIRED = "agent.confirm_required"
AGENT_COMPLETE = "agent.complete"
AGENT_ERROR = "agent.error"
AGENT_STEP = "agent.step"
TOOL_EXECUTE = "tool.execute"
TOOL_RESULT = "tool.result"

# Type aliases
SyncHandler = Callable[..., None]
AsyncHandler = Callable[..., Awaitable[None]]
Handler = SyncHandler | AsyncHandler


class EventBus:
    """Simple publish/subscribe event bus.

    Supports both sync and async handlers. Handlers are called in
    registration order.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Handler) -> Callable[[], None]:
        """Subscribe a handler to an event type.

        Returns a callable that unsubscribes the handler when invoked.
        """
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass  # Already removed

        return unsubscribe

    def emit(self, event_type: str, **data: Any) -> None:
        """Emit an event synchronously.

        Sync handlers are called inline. Async handlers are scheduled
        on the running event loop (if one exists).
        """
        for handler in list(self._handlers.get(event_type, [])):
            try:
                result = handler(**data)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running loop — cannot await; close the coroutine
                        result.close()
            except Exception as e:
                logger.warning(
                    "event_handler_error",
                    event_type=event_type,
                    error=str(e),
                )

    async def emit_async(self, event_type: str, **data: Any) -> None:
        """Emit an event, awaiting async handlers in order."""
        for handler in list(self._handlers.get(event_type, [])):
            try:
                result = handler(**data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(
                    "event_handler_error",
                    event_type=event_type,
                    error=str(e),
                )

    def has_subscribers(self, event_type: str) -> bool:
        """Check if an event type has any subscribers."""
        return bool(self._handlers.get(event_type))

    def clear(self, event_type: str | None = None) -> None:
        """Clear handlers for an event type, or all handlers if None."""
        if event_type is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event_type, None)


# Global event bus singleton
event_bus = EventBus()
