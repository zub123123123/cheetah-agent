import argparse
import logging

from agent.graph import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Агент: поиск фактов в Wikipedia + рассуждение + самопроверка")
    parser.add_argument("question", help="Вопрос на естественном языке")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Подробный лог по узлам графа, HTTP-запросам к Wikipedia и вызовам LLM",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    result = app.invoke({"question": args.question, "retry_count": 0})

    print("=== Chain of thought ===")
    print(result.get("cot", ""))
    print("\n=== Answer ===")
    print(result.get("answer", ""))

    validation = result.get("validation", {})
    retry_count = result.get("retry_count", 0)
    print(f"\n(validation: {validation.get('status')}, retries used: {retry_count})")


if __name__ == "__main__":
    main()
