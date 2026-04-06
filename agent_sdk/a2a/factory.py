"""
Factory for creating A2A Starlette applications.

Eliminates the identical 33-line boilerplate repeated in every agent's
``a2a_service/server.py``. Each agent's server module calls this factory
with its agent-card and executor class, and the factory handles MongoDB
task store creation.

Usage::

    # a2a_service/server.py
    from agent_sdk.a2a.factory import create_a2a_app
    from .agent_card import MY_AGENT_CARD
    from .executor import MyAgentExecutor

    def create_a2a_app_local():
        return create_a2a_app(MY_AGENT_CARD, MyAgentExecutor, "agent_myagent")
"""

import logging
import os

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler

from agent_sdk.a2a.server.mongodb_task_store import AsyncMongoDBTaskStore

logger = logging.getLogger("agent_sdk.a2a.factory")


def create_a2a_app(agent_card, executor_cls, mongo_db_name: str) -> A2AStarletteApplication:
    """Build an A2A Starlette application for the given agent.

    Args:
        agent_card: The ``AgentCard`` instance describing this agent.
        executor_cls: The ``BaseAgentExecutor`` subclass to instantiate.
        mongo_db_name: Default MongoDB database name (overridable via
            ``MONGO_DB_NAME`` env var).

    Returns:
        A configured :class:`A2AStarletteApplication` ready for mounting.
    """
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", mongo_db_name)
    task_store = AsyncMongoDBTaskStore(
        conn_string=mongo_uri,
        db_name=db_name,
        collection_name="a2a_tasks",
    )
    executor = executor_cls()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    logger.info("A2A application created (db='%s')", db_name)
    return a2a_app
