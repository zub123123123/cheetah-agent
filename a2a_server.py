import logging

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from starlette.applications import Starlette

from agent.a2a_agent import CheetahBridgeAgent

HOST = "127.0.0.1"
PORT = 9999
RPC_URL = "/"


def build_agent_card() -> AgentCard:
    skill = AgentSkill(
        id="wiki_fact_reasoning",
        name="Wikipedia fact reasoning",
        description=(
            "Отвечает на вопрос, требующий поиска фактов в Wikipedia, "
            "пошагового рассуждения над ними и самопроверки результата."
        ),
        tags=["wikipedia", "reasoning", "self-check"],
        examples=[
            "Сколько времени потребуется гепарду, чтобы пересечь Москву-реку по Большому Каменному мосту?"
        ],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    )
    return AgentCard(
        name="cheetah-agent",
        description="LangGraph-агент: поиск фактов в Wikipedia + рассуждение + самопроверка",
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=f"http://{HOST}:{PORT}{RPC_URL}",
                protocol_version="1.0",
            )
        ],
        skills=[skill],
    )


def build_app() -> Starlette:
    agent_card = build_agent_card()
    request_handler = DefaultRequestHandler(
        agent_executor=CheetahBridgeAgent(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, RPC_URL))
    return Starlette(routes=routes)


app = build_app()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=HOST, port=PORT)
