import json
import logging
import traceback
from typing import Callable, Awaitable, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    UnsupportedOperationError,
)

from agent_sdk.errors import AgentError, ErrorCode

logger = logging.getLogger("agent_sdk.a2a_executor")


class BaseAgentExecutor(AgentExecutor):
    """
    A generic A2A executor that bridges incoming A2A tasks to an agent's run_query function.
    """

    def __init__(self, run_query_fn: Callable[..., Awaitable[dict[str, Any]]]):
        self.run_query_fn = run_query_fn

    def _extract_kwargs(self, task_metadata: dict) -> dict:
        """
        Extract additional kwargs from task metadata to pass to run_query_fn.
        Override in subclasses to pass specific fields.
        """
        kwargs = {}
        for key in ("mode", "response_format", "model_id", "as_of_date", "watchlist_id"):
            if key in task_metadata:
                kwargs[key] = task_metadata[key]
        return kwargs

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("A2A execute — task_id='%s'", context.task_id)

        query = context.get_user_input()
        if not query:
            logger.error("No text content found in the request")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_FAILED),
                )
            )
            return

        task_metadata = getattr(context, "task", {}).get("metadata", {}) if hasattr(context, "task") else {}
        user_id = task_metadata.get("user_id") or context.context_id or context.task_id
        session_id = context.context_id or context.task_id

        logger.info("A2A execute — task_id='%s', user_id='%s'", context.task_id, user_id)

        kwargs = self._extract_kwargs(task_metadata)

        try:
            result = await self.run_query_fn(query, session_id=session_id, user_id=user_id, **kwargs)
            response_text = result["response"]

            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    artifact=Artifact(
                        artifact_id=f"artifact-{context.task_id}",
                        parts=[Part(text=response_text)],
                    ),
                    last_chunk=True,
                )
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                )
            )
        except Exception as e:
            logger.error("A2A execution failed: %s\n%s", e, traceback.format_exc())
            if isinstance(e, AgentError):
                error_text = json.dumps(e.to_dict())
            else:
                error_text = json.dumps(AgentError(
                    error_code=ErrorCode.INTERNAL,
                    message=str(e),
                ).to_dict())
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_FAILED,
                        message=Message(
                            message_id=f"error-{context.task_id}",
                            role=Role.ROLE_AGENT,
                            parts=[Part(text=error_text)],
                        ),
                    ),
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancel is not supported.")


class StreamingAgentExecutor(BaseAgentExecutor):
    """
    A2A executor that streams response chunks as they are generated.

    Uses an async-generator ``stream_fn`` so the A2A SSE stream carries live
    data (progress markers + text) instead of blocking until the full response
    is ready. This prevents Cloudflare 524 timeouts on long-running queries.

    ``stream_fn`` signature (async generator)::

        async def stream_fn(query, *, session_id, user_id, **kwargs) -> AsyncIterator[str]:
            ...yield chunk...

    Chunks starting with ``__PROGRESS__:`` are forwarded as-is so the
    marketplace can convert them to SSE progress events on the client.
    """

    def __init__(self, run_query_fn, stream_fn):
        super().__init__(run_query_fn=run_query_fn)
        self.stream_fn = stream_fn

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("A2A streaming execute — task_id='%s'", context.task_id)

        query = context.get_user_input()
        if not query:
            logger.error("No text content found in the request")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_FAILED),
                )
            )
            return

        task_metadata = getattr(context, "task", {}).get("metadata", {}) if hasattr(context, "task") else {}
        user_id = task_metadata.get("user_id") or context.context_id or context.task_id
        session_id = context.context_id or context.task_id

        logger.info("A2A streaming execute — task_id='%s', user_id='%s'", context.task_id, user_id)

        kwargs = self._extract_kwargs(task_metadata)
        artifact_id = f"artifact-{context.task_id}"

        try:
            async for chunk in self.stream_fn(query, session_id=session_id, user_id=user_id, **kwargs):
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        task_id=context.task_id,
                        context_id=context.context_id,
                        artifact=Artifact(
                            artifact_id=artifact_id,
                            parts=[Part(text=chunk)],
                        ),
                        last_chunk=False,
                    )
                )

            # Signal end of stream
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    artifact=Artifact(
                        artifact_id=artifact_id,
                        parts=[Part(text="")],
                    ),
                    last_chunk=True,
                )
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                )
            )
        except Exception as e:
            logger.error("A2A streaming execution failed: %s\n%s", e, traceback.format_exc())
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_FAILED,
                        message=Message(
                            message_id=f"error-{context.task_id}",
                            role=Role.ROLE_AGENT,
                            parts=[Part(text=str(e))],
                        ),
                    ),
                )
            )
