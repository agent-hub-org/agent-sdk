"""Shared SSE event-stream generator used by all agent app.py files."""
import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable, Awaitable

logger = logging.getLogger("agent_sdk.server.sse")

_HEARTBEAT_INTERVAL = 15.0
_PROGRESS_PREFIX = "__PROGRESS__:"
_ERROR_PREFIX = "__ERROR__:"


async def create_sse_stream(
    stream,
    session_id: str,
    query: str,
    on_complete: Callable[[str, list, str | None], Awaitable[None]] | None = None,
) -> AsyncGenerator[str, None]:
    """Wrap an agent StreamResult as an SSE generator.

    Args:
        stream: A StreamResult (or async-iterable) from agent.astream().
        session_id: Current session identifier, emitted in the final event.
        query: Original user query (passed through to on_complete for logging).
        on_complete: Async callback(response_text, steps, plan) called after
                     the stream ends — use it to save conversations / memories.
    """
    _stream_timeout = float(os.getenv("STREAM_TIMEOUT_SECONDS", "300"))
    full_response: list[str] = []
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=100)

    async def _heartbeat():
        try:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
                await queue.put(f": heartbeat {int(asyncio.get_running_loop().time())}\n\n")
        except asyncio.CancelledError:
            pass

    async def _producer():
        try:
            async with asyncio.timeout(_stream_timeout):
                async for chunk in stream:
                    try:
                        await asyncio.wait_for(queue.put(chunk), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning("SSE queue full for session='%s' — client likely disconnected", session_id)
                        return
        except TimeoutError:
            logger.error("Stream timed out after %.0fs for session='%s'", _stream_timeout, session_id)
            await queue.put(f"{_ERROR_PREFIX}Response timed out after {_stream_timeout:.0f} seconds.")
        except Exception as exc:
            logger.error("Stream producer failed for session='%s': %s", session_id, exc)
            await queue.put(f"{_ERROR_PREFIX}An internal error occurred while generating the response.")
        finally:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    heartbeat_task = asyncio.create_task(_heartbeat())
    producer_task = asyncio.create_task(_producer())

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break

            if isinstance(chunk, str):
                if chunk.startswith(": heartbeat"):
                    yield chunk
                elif chunk.startswith(_PROGRESS_PREFIX):
                    phase_label = chunk[len(_PROGRESS_PREFIX):]
                    yield f"event: progress\ndata: {json.dumps({'phase': phase_label})}\n\n"
                elif chunk.startswith(_ERROR_PREFIX):
                    error_msg = chunk[len(_ERROR_PREFIX):]
                    yield f"event: error\ndata: {json.dumps({'message': error_msg})}\n\n"
                    fallback = f"\n\n[{error_msg}]"
                    yield f"data: {json.dumps({'text': fallback})}\n\n"
                    full_response.append(fallback)
                else:
                    full_response.append(chunk)
                    yield f"data: {json.dumps({'text': chunk})}\n\n"

        response_text = "".join(full_response)
        if not response_text.strip():
            response_text = "Sorry, the model returned an empty response. Please try again."
            yield f"data: {json.dumps({'text': response_text})}\n\n"

        if on_complete is not None:
            try:
                steps = stream.steps if hasattr(stream, "steps") else []
                plan = stream.plan if hasattr(stream, "plan") else None
                await on_complete(response_text, steps, plan)
            except Exception as exc:
                logger.error("on_complete failed for session='%s': %s", session_id, exc)

        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
    finally:
        heartbeat_task.cancel()
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        yield "data: [DONE]\n\n"
