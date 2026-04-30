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

from starlette.applications import Starlette
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_rest_routes

from agent_sdk.a2a.server.mongodb_task_store import AsyncMongoDBTaskStore

logger = logging.getLogger("agent_sdk.a2a.factory")


class _A2AApp(Starlette):
    """Starlette app that exposes .build() for backward compat with agent app.py files."""

    def build(self) -> Starlette:
        return self


def create_a2a_app(agent_card, executor_cls, mongo_db_name: str) -> _A2AApp:
    """Build an A2A Starlette application for the given agent.

    Args:
        agent_card: The ``AgentCard`` instance describing this agent.
        executor_cls: The ``BaseAgentExecutor`` subclass to instantiate.
        mongo_db_name: Default MongoDB database name (overridable via
            ``MONGO_DB_NAME`` env var).

    Returns:
        A configured :class:`_A2AApp` (Starlette subclass) ready for mounting.
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
        agent_card=agent_card,
    )
    routes = create_agent_card_routes(agent_card) + create_rest_routes(request_handler)
    app = _A2AApp(routes=routes)
    logger.info("A2A application created (db='%s')", db_name)
    return app
