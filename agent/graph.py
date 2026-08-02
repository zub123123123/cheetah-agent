import json
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .config import MAX_RETRIES
from .llm import call_llm_json
from .tools import fetch_article_text, search_wikipedia

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    search_queries: list[dict]
    raw_articles: list[dict]
    facts: list[dict]
    cot: str
    answer: str
    validation: dict
    retry_count: int


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 2,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "lang": {"type": "string", "enum": ["ru", "en"]},
                },
                "required": ["query", "lang"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "property": {"type": "string"},
                    "value": {"type": "string"},
                    "unit": {"type": ["string", "null"]},
                    "source_text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "entity",
                    "property",
                    "value",
                    "unit",
                    "source_text",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

REASON_SCHEMA = {
    "type": "object",
    "properties": {
        "cot": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["cot", "answer"],
    "additionalProperties": False,
}

VALIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "status": {"type": "string", "enum": ["VALID", "INVALID"]},
    },
    "required": ["reason", "status"],
    "additionalProperties": False,
}


def search_node(state: AgentState) -> dict:
    logger.info("search_node: start (retry_count=%s)", state.get("retry_count", 0))
    system = (
        "Ты превращаешь вопрос в 2-5 поисковых запросов для Wikipedia, нужных, чтобы найти "
        "факты для ответа на него. Каждый запрос должен быть точным или почти точным "
        "названием статьи Wikipedia об этой сущности (например, «Большой Каменный мост», "
        "«Cheetah») — а не описательной поисковой фразой с лишними словами вроде «длина» или "
        "«максимальная скорость», потому что такие лишние слова заставляют поиск Wikipedia "
        "ранжировать выше не относящиеся к делу статьи. Для каждого запроса выбирай язык "
        "(ru или en), на котором статья с большей вероятностью существует и подробнее раскрыта."
    )
    user = f"Вопрос: {state['question']}"

    validation = state.get("validation")
    if validation and validation.get("status") == "INVALID":
        user += (
            f"\n\nПредыдущая попытка была отклонена на валидации.\n"
            f"Причина: {validation['reason']}\n"
            f"Предыдущие поисковые запросы: {json.dumps(state.get('search_queries', []), ensure_ascii=False)}\n"
            f"Предыдущие запросы либо не нашли нужную статью, либо привели не к той. "
            f"Предложи другие запросы — попробуй альтернативные названия, синонимы или другой "
            f"язык — чтобы действительно найти факты, указанные в причине отклонения."
        )

    result = call_llm_json(system, user, SEARCH_SCHEMA, "search_queries")
    logger.info("search_node: queries=%s", result["queries"])
    return {"search_queries": result["queries"]}


def wiki_node(state: AgentState) -> dict:
    logger.info("wiki_node: start, %d queries", len(state["search_queries"]))
    articles = []
    for q in state["search_queries"]:
        query, lang = q["query"], q["lang"]
        titles = search_wikipedia(query, lang)
        for title in titles[:2]:
            text = fetch_article_text(title, lang)
            if text:
                articles.append({"title": title, "lang": lang, "text": text})
    logger.info("wiki_node: done, %d articles fetched", len(articles))
    return {"raw_articles": articles}


def extract_node(state: AgentState) -> dict:
    logger.info("extract_node: start")
    articles = state.get("raw_articles", [])
    if articles:
        context = "\n\n".join(
            f"[{a['lang']}] {a['title']}:\n{a['text']}" for a in articles
        )
    else:
        context = "(статьи не найдены)"

    system = (
        "Извлеки структурированные факты (entity, property, value, unit, source_text, "
        "confidence) из приведённых ниже фрагментов статей Wikipedia, относящиеся к ответу "
        "на вопрос. Включай только факты, которые правдоподобно могут помочь ответить на "
        "вопрос. Используй пустую строку для unit, если у значения нет единицы измерения. "
        "Будь точен в поле `entity` в том, к какой версии/эпохе объекта относится значение. "
        "Статьи Wikipedia часто описывают более раннего предшественника, прежнее название "
        "или состояние, которое позже было заменено, в том же тексте, где обсуждается "
        "современный объект (обращай внимание на явные даты или слова вроде «ранее», «прежде», "
        "«первоначально», «до перестройки/переименования/замены», «предшественник» и их "
        "эквиваленты на других языках). Если значение явно относится к такому более раннему "
        "состоянию, всё равно извлеки его, но явно укажи это в `entity` (например, «X "
        "(исторический предшественник, XVII век)»), а не называй его так же, как современный "
        "объект — никогда не объединяй историческое значение под тем же `entity`, что и "
        "сегодняшнюю версию объекта."
    )
    user = f"Вопрос: {state['question']}\n\nСтатьи:\n{context}"

    result = call_llm_json(system, user, EXTRACT_SCHEMA, "facts")
    logger.info("extract_node: extracted %d facts", len(result["facts"]))
    return {"facts": result["facts"]}


def reason_node(state: AgentState) -> dict:
    logger.info("reason_node: start")
    system = (
        "Рассуждай пошагово (chain of thought), используя только предоставленные факты, "
        "чтобы ответить на вопрос. Выполняй все необходимые вычисления прямо в тексте "
        "рассуждения. Если какого-то факта не хватает, явно скажи об этом в рассуждении."
    )
    user = (
        f"Вопрос: {state['question']}\n\n"
        f"Факты:\n{json.dumps(state.get('facts', []), ensure_ascii=False)}"
    )

    result = call_llm_json(system, user, REASON_SCHEMA, "reasoning")
    logger.info("reason_node: answer=%r", result["answer"])
    return {"cot": result["cot"], "answer": result["answer"]}


def validate_node(state: AgentState) -> dict:
    logger.info("validate_node: start")
    system = (
        "Ты независимо проверяешь ответ на вопрос. В поле `reason` пошагово, как "
        "chain-of-thought, пересчитай результат с нуля, используя только данные факты, не "
        "доверяя предоставленному рассуждению. Проверь: "
        "(1) все ли факты использованы правильно, (2) совпадает ли пересчёт с ответом, "
        "(3) согласованы ли единицы измерения, (4) не упущено ли ограничение, которое меняет "
        "применимость фактов именно к этому случаю, (5) действительно ли каждый факт описывает "
        "текущее, современное состояние сущности из вопроса — а не более раннюю версию, "
        "предшественника, прежнее название или значение, которое было позже заменено и просто "
        "упомянуто в том же исходном тексте (обращай внимание на явные даты или слова вроде "
        "«ранее», «прежде», «первоначально», «до перестройки/переименования/замены», "
        "«предшественник» и их эквиваленты на других языках). Если использованный факт на "
        "самом деле описывает прошлое состояние, а не настоящее, ответ INVALID, независимо от "
        "того, верна ли арифметика. Только закончив это пошаговое рассуждение, установи "
        "`status` как вывод, к которому пришёл твой собственный `reason` — `status` должен "
        "следовать из `reason`, а не наоборот."
    )
    user = (
        f"Вопрос: {state['question']}\n"
        f"Факты: {json.dumps(state.get('facts', []), ensure_ascii=False)}\n"
        f"Рассуждение: {state.get('cot', '')}\n"
        f"Ответ: {state.get('answer', '')}"
    )

    result = call_llm_json(system, user, VALIDATE_SCHEMA, "validation")

    retry_count = state.get("retry_count", 0)
    if result["status"] == "INVALID":
        retry_count += 1

    logger.info(
        "validate_node: status=%s retry_count=%d reason=%r",
        result["status"],
        retry_count,
        result["reason"],
    )
    return {"validation": result, "retry_count": retry_count}


def _route_after_validate(state: AgentState) -> str:
    if state["validation"]["status"] == "INVALID" and state["retry_count"] < MAX_RETRIES:
        return "retry"
    return "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("search", search_node)
    graph.add_node("wiki", wiki_node)
    graph.add_node("extract", extract_node)
    graph.add_node("reason", reason_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "wiki")
    graph.add_edge("wiki", "extract")
    graph.add_edge("extract", "reason")
    graph.add_edge("reason", "validate")

    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"retry": "search", "end": END},
    )

    return graph.compile()


app = build_graph()
