import logging
import traceback
from typing import Callable, Awaitable, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)

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
        if "mode" in task_metadata:
            kwargs["mode"] = task_metadata["mode"]
        if "response_format" in task_metadata:
            kwargs["response_format"] = task_metadata["response_format"]
        if "model_id" in task_metadata:
            kwargs["model_id"] = task_metadata["model_id"]
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
                    final=True,
                    status=TaskStatus(state=TaskState.failed),
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
                        parts=[Part(root=TextPart(text=response_text))],
                    ),
                    last_chunk=True,
                )
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    final=True,
                    status=TaskStatus(state=TaskState.completed),
                )
            )
        except Exception as e:
            logger.error("A2A execution failed: %s\n%s", e, traceback.format_exc())
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    final=True,
                    status=TaskStatus(state=TaskState.failed),
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancel is not supported.")
