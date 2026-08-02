import json
import logging
import time

from openai import OpenAI

from .config import API_KEY, BASE_URL, LLM_TIMEOUT_SECONDS, MODEL_NAME

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=LLM_TIMEOUT_SECONDS)
    return _client


def call_llm_json(system: str, user: str, schema: dict, schema_name: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
    }
    logger.info("LLM request %r -> endpoint=%s model=%s", schema_name, BASE_URL, MODEL_NAME)
    logger.debug("LLM request %r payload: %s", schema_name, json.dumps(payload, ensure_ascii=False))

    start = time.perf_counter()
    try:
        response = _get_client().chat.completions.create(**payload)
    except Exception:
        logger.exception("LLM request %r failed after %.2fs", schema_name, time.perf_counter() - start)
        raise
    elapsed = time.perf_counter() - start

    logger.info(
        "LLM response %r received in %.2fs (finish_reason=%s)",
        schema_name,
        elapsed,
        response.choices[0].finish_reason,
    )
    logger.debug("LLM response %r raw: %s", schema_name, response.model_dump_json())

    content = response.choices[0].message.content
    return json.loads(content)
