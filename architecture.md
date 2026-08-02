# Архитектура агента (тестовое задание)

Агент отвечает на произвольный вопрос, требующий: (1) поиска фактов во внешней среде (Wikipedia), (2) рассуждения над ними, (3) самопроверки результата. Пример из задания — время пересечения Москвы-реки гепардом по Большому Каменному мосту — не зашит в код: логика полностью общая.

**Фреймворк:** LangGraph
**Стиль:** жёсткий линейный пайплайн (не ReAct-цикл) + одна условная петля самокоррекции при невалидном результате

---

## 1. Workflow

```
question
   │
   ▼
┌─────────┐   search queries (с языком: ru/en)
│ search  │──────────────────────────────┐
└─────────┘                               │
   ▲                                      ▼
   │                                 ┌─────────┐
   │                                 │  wiki   │  Wikipedia API: search + fetch
   │                                 └─────────┘
   │                                      │ raw article texts
   │                                      ▼
   │                                 ┌─────────┐
   │                                 │ extract │  LLM → факты в JSON
   │                                 └─────────┘
   │                                      │ facts[]
   │                                      ▼
   │                                 ┌─────────┐
   │                                 │ reason  │  LLM → CoT + answer
   │                                 └─────────┘        (вычисления прямо в рассуждении,
   │                                      │               без отдельного calculator-tool)
   │                                      ▼
   │                                 ┌──────────┐
   │                                 │ validate │  независимый пересчёт "с нуля",
   │                                 └──────────┘  проверка фактов, единиц, допущений
   │                                      │
   │              INVALID (retry_count < MAX)     VALID  или  retry_count >= MAX
   └──────────────────────────────────────┘              │
                                                           ▼
                                                          END
```

Узлы — обычные функции `(state) -> state`, склеенные `StateGraph`. Никакой доменной логики про гепардов/мосты в коде — всё "знание" живёт в промптах.

---

## 2. State

```python
class AgentState(TypedDict):
    question: str
    search_queries: list[dict]      # [{"query": "...", "lang": "ru"|"en"}]
    raw_articles: list[dict]        # [{"title", "lang", "text"}]
    facts: list[dict]               # [{"entity","property","value","unit","source_text","confidence"}]
    cot: str
    answer: str
    validation: dict                # {"status": "VALID"|"INVALID", "reason": str}
    retry_count: int
```

---

## 3. Узлы

**search_node** — LLM превращает вопрос в 1-3 поисковых запроса, каждый с указанием языка (ru/en) — модель сама решает, на каком языке вероятнее найти нужный факт. При повторном заходе (после INVALID) получает на вход причину отказа и корректирует запросы.

**wiki_node** — по каждому запросу: `search_wikipedia(query, lang)` → кандидаты статей → `fetch_article_text(title, lang)` → тексты (обрезка до ~3000 символов). Fallback: если статья не найдена по точному названию — повторный поиск по ключевым словам.

**extract_node** — LLM извлекает факты из объединённого текста статей в структурированный JSON (`entity, property, value, unit, source_text`).

**reason_node** — LLM строит пошаговое рассуждение (CoT) на основе фактов, включая любые вычисления прямо в тексте рассуждения (моделям современного уровня достаточно точности для такой арифметики — отдельный calculator-tool не добавляем), и даёт финальный ответ. Возвращает `{"cot": "...", "answer": "..."}`.

**validate_node** — независимая (другой промпт, отдельный вызов LLM) проверка: (1) все ли факты корректно использованы, (2) пересчёт "с нуля", (3) согласованность единиц измерения, (4) не упущено ли ограничение, существенно меняющее применимость использованных величин к конкретному случаю. Возвращает `{"status": "VALID"|"INVALID", "reason": "..."}`. При INVALID и `retry_count < MAX_RETRIES` — граф возвращается в `search_node` с накопленной в state причиной отказа.

---

## 4. Граф (LangGraph)

```python
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
    lambda s: "retry" if s["validation"]["status"] == "INVALID" and s["retry_count"] < MAX_RETRIES else "end",
    {"retry": "search", "end": END},
)

app = graph.compile()
```

---

## 5. A2A-обёртка (опционально)

```python
class CheetahBridgeAgent:
    async def execute(self, task: Task) -> Task:
        question = extract_question(task.messages)
        result = await asyncio.to_thread(app.invoke, {
            "question": question,
            "retry_count": 0,
        })
        return build_task_response(result["answer"], result["cot"])
```

Тонкий адаптер поверх скомпилированного графа — вся логика остаётся в `app`.

---

## 6. Структура репозитория

```
.
├── agent/
│   ├── __init__.py
│   ├── config.py            # OPENAI_API_KEY, MODEL_NAME, WIKI_USER_AGENT
│   ├── tools.py              # search_wikipedia(), fetch_article_text()
│   ├── llm.py                 # call_llm()
│   ├── graph.py               # AgentState, узлы, сборка StateGraph
│   └── a2a_agent.py           # CheetahBridgeAgent (опционально)
├── main.py                    # CLI: python main.py "вопрос"
├── requirements.txt            # langgraph, openai, wikipedia-api, (google-a2a)
├── .env.example
└── README.md                   # описание, схема workflow, пример запуска
```

---

## 7. Ключевые решения и почему

| Решение | Почему |
|---|---|
| Жёсткий пайплайн, а не ReAct-цикл | Проще, предсказуемее, легче проверять; explicit CoT в reason_node всё равно даёт видимое "рассуждение" |
| LangGraph | Осознанный выбор фреймворка, узлы = чистые функции, граф явно описывает workflow |
| Retry-петля validate→search | Общий механизм отказоустойчивости (Wikipedia может не дать нужный факт с первого раза), не завязан на предметную область |
| Без отдельного calculator-tool | Простая арифметика — не проблема для современных моделей; лишний tool call не даёт функциональной пользы |
| Self-check "с нуля" в отдельном вызове | Независимый пересчёт эффективнее ловит ошибки, чем модель, соглашающаяся сама с собой |
| Многоязычный поиск (ru/en) | Разные факты вероятнее находятся в разных языковых версиях Wikipedia |
