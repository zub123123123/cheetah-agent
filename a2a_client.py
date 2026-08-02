import argparse
import asyncio
import uuid

import httpx
from a2a.client import ClientConfig, create_client
from a2a.types import Message, Part, Role, a2a_pb2

DEFAULT_URL = "http://127.0.0.1:9999"
DEFAULT_TIMEOUT_SECONDS = 300.0


async def ask(server_url: str, question: str, timeout_seconds: float) -> None:
    httpx_client = httpx.AsyncClient(timeout=timeout_seconds)
    client = await create_client(server_url, client_config=ClientConfig(httpx_client=httpx_client))
    try:
        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.ROLE_USER,
            parts=[Part(text=question)],
        )
        request = a2a_pb2.SendMessageRequest(message=message)
        async for response in client.send_message(request):
            answer = "".join(part.text for part in response.message.parts if part.text)
            metadata = dict(response.message.metadata)

            print("=== Chain of thought ===")
            print(metadata.get("cot", ""))
            print("\n=== Answer ===")
            print(answer)
            print(
                f"\n(validation: {metadata.get('validation_status')}, "
                f"retries used: {int(metadata.get('retry_count', 0))})"
            )
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A2A-клиент: отправляет вопрос агенту, поднятому через a2a_server.py"
    )
    parser.add_argument("question", help="Вопрос на естественном языке")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Адрес A2A-сервера (по умолчанию {DEFAULT_URL})")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP-таймаут ожидания ответа в секундах (по умолчанию {DEFAULT_TIMEOUT_SECONDS:.0f} — "
        "полный прогон графа с ретраями может занимать несколько минут)",
    )
    args = parser.parse_args()

    asyncio.run(ask(args.url, args.question, args.timeout))


if __name__ == "__main__":
    main()
