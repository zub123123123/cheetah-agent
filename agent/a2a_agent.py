import asyncio
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, Role, UnsupportedOperationError
from google.protobuf.struct_pb2 import Struct

from .graph import app


class CheetahBridgeAgent(AgentExecutor):
    """Тонкий A2A-адаптер поверх скомпилированного LangGraph-графа `app`."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = context.get_user_input()

        result = await asyncio.to_thread(app.invoke, {"question": question, "retry_count": 0})

        metadata = Struct()
        metadata.update(
            {
                "cot": result.get("cot", ""),
                "validation_status": result.get("validation", {}).get("status", ""),
                "retry_count": result.get("retry_count", 0),
            }
        )

        message = Message(
            message_id=str(uuid.uuid4()),
            context_id=context.context_id,
            task_id=context.task_id,
            role=Role.ROLE_AGENT,
            parts=[Part(text=result.get("answer", ""))],
            metadata=metadata,
        )
        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError()
