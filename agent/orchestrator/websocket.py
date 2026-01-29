"""WebSocket connection management and endpoints.

This module contains:
- ConnectionManager for WebSocket connections
- WebSocket endpoints for real-time updates
"""

import asyncio
import contextlib
import logging
import uuid
from collections import defaultdict
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from agent.config import settings
from agent.llm.client import LLMError, call_llm_chat_stream_async
from agent.orchestrator.deps import get_sandbox, get_task_manager
from agent.orchestrator.models import WebSocketMessage, WSMessageType
from agent.orchestrator.react_agent import ReActAgent
from agent.orchestrator.session import add_message, get_history
from agent.orchestrator.task_manager import TaskState as TMState

logger = logging.getLogger(__name__)

# Shared resources
sandbox = get_sandbox()
task_manager = get_task_manager()


class ConnectionManager:
    """Manages WebSocket connections and task subscriptions."""

    def __init__(self):
        self.connections: set[WebSocket] = set()
        self.task_subs: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, ws: WebSocket):
        """Accept a new WebSocket connection."""
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket):
        """Remove a WebSocket connection."""
        self.connections.discard(ws)
        for subs in self.task_subs.values():
            subs.discard(ws)

    def subscribe(self, ws: WebSocket, task_id: str):
        """Subscribe a connection to task updates."""
        self.task_subs[task_id].add(ws)

    async def broadcast(self, task_id: str, msg: WebSocketMessage | dict):
        """Broadcast a message to all subscribers of a task."""
        dead = set()
        data = msg.model_dump() if isinstance(msg, WebSocketMessage) else msg
        for conn in self.task_subs.get(task_id, set()):
            try:
                await conn.send_json(data)
            except Exception:
                dead.add(conn)
        for conn in dead:
            self.disconnect(conn)

    async def broadcast_all(self, msg: WebSocketMessage):
        """Broadcast a typed message to all connected clients."""
        dead = set()
        for conn in self.connections:
            try:
                await conn.send_json(msg.model_dump())
            except Exception:
                dead.add(conn)
        for conn in dead:
            self.disconnect(conn)

    async def send_step_output(self, task_id: str, step: str, output: Any):
        """Send step output to task subscribers."""
        await self.broadcast(
            task_id, WebSocketMessage.step_output(task_id, step, output)
        )

    async def send_task_complete(self, task_id: str, summary: str):
        """Notify subscribers that a task completed."""
        await self.broadcast(task_id, WebSocketMessage.task_complete(task_id, summary))

    async def send_task_error(self, task_id: str, error: str):
        """Notify subscribers of a task error."""
        await self.broadcast(task_id, WebSocketMessage.task_error(task_id, error))


# Global WebSocket manager
ws_manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time task updates.

    Supported messages:
    - {"type": "subscribe", "task_id": "..."} - Subscribe to task updates
    - {"type": "unsubscribe", "task_id": "..."} - Unsubscribe from task
    - {"type": "ping"} - Keep-alive ping
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            raw_data = await websocket.receive_json()
            try:
                msg = WebSocketMessage.model_validate(raw_data)

                if msg.type == WSMessageType.SUBSCRIBE:
                    if msg.task_id:
                        ws_manager.subscribe(websocket, msg.task_id)
                        response = WebSocketMessage.subscribed(msg.task_id)
                        await websocket.send_json(response.model_dump())
                    else:
                        error = WebSocketMessage.error("task_id required for subscribe")
                        await websocket.send_json(error.model_dump())

                elif msg.type == WSMessageType.UNSUBSCRIBE:
                    if msg.task_id:
                        ws_manager.task_subs.get(msg.task_id, set()).discard(websocket)
                        await websocket.send_json(
                            {"type": "unsubscribed", "task_id": msg.task_id}
                        )

                elif msg.type == WSMessageType.PING:
                    response = WebSocketMessage.pong()
                    await websocket.send_json(response.model_dump())

            except ValidationError as e:
                error = WebSocketMessage.error(
                    f"Invalid message format: {e.error_count()} errors"
                )
                await websocket.send_json(error.model_dump())

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


async def stream_task(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for streaming task execution with real-time token delivery.

    This provides a more responsive experience by streaming:
    - Agent thoughts as they're generated
    - Actions being taken
    - Tool outputs
    - Final response tokens

    Usage:
        1. Connect to ws://host/ws/stream/{task_id}
        2. Send: {"request": "your task", "session_id": "optional"}
        3. Receive streaming messages
    """
    await websocket.accept()

    try:
        # Wait for the task request
        data = await websocket.receive_json()
        request_text = data.get("request", "")
        session_id = data.get("session_id") or str(uuid.uuid4())

        if not request_text:
            await websocket.send_json(
                WebSocketMessage.error("No request provided").model_dump()
            )
            await websocket.close()
            return

        add_message(session_id, "user", request_text)

        # Create task
        task = task_manager.create_task(request_text, session_id)
        task_manager.update_state(task.id, TMState.EXECUTING)

        # Send stream start
        await websocket.send_json(
            WebSocketMessage.stream_start(task.id, "agent").model_dump()
        )

        async def on_progress(iteration: int, status: str, thought: str, action: str):
            """Stream progress updates."""
            try:
                await websocket.send_json(
                    WebSocketMessage.stream_thought(
                        task.id, thought, iteration
                    ).model_dump()
                )
                if action:
                    await websocket.send_json(
                        {
                            "type": "progress",
                            "task_id": task.id,
                            "iteration": iteration,
                            "status": status,
                            "action": action,
                        }
                    )
            except Exception:
                pass  # Connection may have closed

        async def on_confirm(command: str, reason: str, message: str) -> bool:
            """Request confirmation via WebSocket."""
            confirm_id = str(uuid.uuid4())

            await websocket.send_json(
                {
                    "type": "confirm_request",
                    "confirm_id": confirm_id,
                    "task_id": task.id,
                    "command": command[:200],
                    "reason": reason,
                    "message": message,
                }
            )

            try:
                # Wait for confirmation response
                response = await asyncio.wait_for(
                    websocket.receive_json(), timeout=60.0
                )
                return response.get("confirmed", False)
            except TimeoutError:
                return False

        history = get_history(session_id)
        conv = [{"role": m.role, "content": m.content} for m in history]

        agent = ReActAgent(
            sandbox=sandbox,
            on_progress=on_progress,
            on_confirm=on_confirm,
            max_iterations=settings.max_agent_iterations,
            conversation_history=conv,
            require_confirmation=True,
        )

        state = await agent.run(request_text)

        # Update task state
        if state.status == "completed":
            task_manager.update_state(task.id, TMState.COMPLETED)
            task_manager.set_summary(task.id, state.final_answer or "Done")
        else:
            task_manager.update_state(task.id, TMState.FAILED, state.error)

        if state.final_answer:
            add_message(session_id, "assistant", state.final_answer)

            # Stream the final response token by token for a nice effect
            await websocket.send_json(
                WebSocketMessage.stream_start(task.id, "response").model_dump()
            )

            # Send response in chunks for streaming effect
            response = state.final_answer
            chunk_size = 10  # Characters per chunk
            for i in range(0, len(response), chunk_size):
                chunk = response[i : i + chunk_size]
                await websocket.send_json(
                    WebSocketMessage.stream_token(task.id, chunk).model_dump()
                )
                await asyncio.sleep(0.02)  # Small delay for visual effect

            await websocket.send_json(
                WebSocketMessage.stream_end(task.id, response).model_dump()
            )

        # Send completion
        await websocket.send_json(
            {
                "type": "complete",
                "task_id": task.id,
                "session_id": session_id,
                "status": state.status,
                "response": state.final_answer,
                "steps": len(state.steps),
            }
        )

    except WebSocketDisconnect:
        logger.info(f"Stream client disconnected for task {task_id}")
    except LLMError as e:
        await websocket.send_json(
            WebSocketMessage.task_error(task_id, str(e)).model_dump()
        )
    except Exception as e:
        logger.exception(f"Stream error for task {task_id}")
        with contextlib.suppress(Exception):
            await websocket.send_json(WebSocketMessage.error(str(e)).model_dump())
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


async def stream_chat(websocket: WebSocket):
    """
    Simple streaming chat WebSocket - streams LLM responses token by token.

    For quick conversational interactions without full agent execution.

    Usage:
        1. Connect to ws://host/ws/chat
        2. Send: {"message": "hello", "session_id": "optional"}
        3. Receive streaming tokens
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            session_id = data.get("session_id", session_id)

            if not message:
                continue

            add_message(session_id, "user", message)
            history = get_history(session_id)
            messages = [{"role": m.role, "content": m.content} for m in history]

            # Start streaming
            await websocket.send_json(
                {
                    "type": "stream_start",
                    "session_id": session_id,
                }
            )

            full_response = ""
            try:
                async for token in call_llm_chat_stream_async(messages):
                    full_response += token
                    await websocket.send_json(
                        {
                            "type": "stream_token",
                            "token": token,
                        }
                    )

                await websocket.send_json(
                    {
                        "type": "stream_end",
                        "full_response": full_response,
                        "session_id": session_id,
                    }
                )

                add_message(session_id, "assistant", full_response)

            except LLMError as e:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": str(e),
                    }
                )

    except WebSocketDisconnect:
        logger.info("Chat stream client disconnected")
    except Exception as e:
        logger.exception("Chat stream error")
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(e)})
