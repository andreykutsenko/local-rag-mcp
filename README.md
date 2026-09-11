# Local RAG/MCP Knowledge Base Assistant — гибридный поиск

Форк [MobilaName/local-rag-mcp](https://github.com/MobilaName/local-rag-mcp),
домашнее задание 5 курса: гибридный поиск (вектор + полнотекстовый), слияние
выдач по RRF и извлечение ключевых слов локальной моделью, с замерами
«до» и «после» на замороженном наборе вопросов.

- Техническое задание: `SPEC-hybrid.md`; оригинал задания автора курса: `docs/Task-RAG.md`.
- Результаты, предсказания и разбор: `REPORT-hybrid.md`.

## Что это за проект

Локальная система вопросов и ответов по документации организации. Всё
работает на машине пользователя: документы не покидают диск, внешних API нет.

- **RAG**: документы режутся на чанки, чанки индексируются, на вопрос
  подбирается контекст, ответ генерирует локальная модель через Ollama.
- **MCP**: сервер инструментов `read_document`, `list_documents`,
  `search_documents`; модель может запросить полный документ.
- **CLI**: `python main.py` — интерактивный режим.

```
вопрос → ключевые слова (Ollama) → [вектор FAISS ‖ FTS5 SQLite] → RRF → топ-5
       → промпт с контекстом → Ollama → ответ + источники
```

Стек: Python 3.10+, FAISS, SentenceTransformers (`all-MiniLM-L6-v2`),
SQLite FTS5 (стандартная библиотека), Ollama (`qwen3:0.6b`), FastMCP.

## Что добавлено в форке

| Модуль | Назначение |
|---|---|
| `src/rag/keywords.py` | `extract_keywords(question)`: один вызов модели, `temperature=0`, `think: false`, терпимый парсер, fallback на исходный вопрос |
| `src/rag/fulltext.py` | `build_fts(chunks)` и `search_fts(query, top_k)`: SQLite FTS5 по тем же чанкам и с теми же идентификаторами, что FAISS |
| `src/rag/fusion.py` | `reciprocal_rank_fusion(runs, k=60, weights=None)`: слияние ранжированных списков, о поиске не знает |
| `src/rag/hybrid.py` | оркестрация: ключевые слова → вектор и FTS параллельно → RRF → топ-5 → ответ модели |
| `src/rag/query.py` | новая `search_vector(query, top_k)` для слияния; `retrieve` не менялась; в `ask_llm` добавлен `think: false` |
| `src/rag/build_index.py` | строит FTS-индекс рядом с FAISS |
| `src/rag/ingest.py` | исправлен сбой апстрима на вложенных каталогах документов |
| `src/assistant.py` | контекст берётся через `retrieve_hybrid` |
| `bench/` | `tasks.py` (замороженный набор вопросов), `metrics.py`, `run.py`, `report.py`, `results/*.json` |
| `tests/` | 31 тест по разделу `<tests>` спеки, внешних вызовов нет |

Решения, принятые по спеке без уточняющих вопросов:

- **Query Expansion — ключевые слова**, а не альтернативные формулировки.
  В поиск уходит одна строка: вопрос, перевод строки, ключевые слова через
  запятую (`expand_query`), одинаковая для вектора и FTS.
- **Парсер ответа модели терпимый**: режет по запятым и переводам строк,
  снимает маркеры списков и эхо метки `Keywords:`, убирает пустые и дубликаты
  без учёта регистра, выбрасывает элементы длиннее 60 символов. Fallback на
  исходный вопрос только если ничего не разобралось или ответ длиннее
  200 символов; предупреждение в лог, прогон продолжается.
- **Запрос к FTS**: каждое слово в кавычках, между словами `OR`, чтобы
  пунктуация и `--force-with-lease` не ломали синтаксис MATCH; ранжирует bm25.
- **Глубина кандидатов** для слияния: 20 на каждый поиск, топ-5 после RRF.
  Выбрано до замера, не подбиралось.
- **Веса RRF**: параметр `weights`, по умолчанию `(1.0, 1.0)`. Вес 0.3 для
  вектора проверен одним отдельным замером как гипотеза, не подбирался.
- **Параллельность**: два блокирующих поиска запускаются в
  `ThreadPoolExecutor`, пул открывается на вызов и закрывается контекстным
  менеджером. Отказ FTS не роняет запрос: предупреждение в лог, выдача из
  векторного списка. Отказ вектора поднимается наверх.
- **Метрика ранга**: позиция первого чанка из допустимого файла в топ-5.
- **Ollama**: `think: false` во всех запросах (без этого qwen3 пишет блок
  рассуждений и на CPU не укладывается в минуты); `num_thread` не передаётся,
  значение 6 зашито в модель на этой машине, своё перебило бы его.

## Окружение с нуля

Требования: Python 3.10+, запущенная Ollama с моделью `qwen3:0.6b`.

```bash
# 1. Виртуальное окружение в корне проекта и зависимости
python3 -m venv .venv                       # если нет ensurepip: uv venv .venv --seed
.venv/bin/pip install -r src/requirements.txt   # или: uv pip install --python .venv/bin/python -r src/requirements.txt
.venv/bin/pip install pytest                    # только для тестов

# 2. Ollama: сервер отвечает, модель на месте
curl -s http://127.0.0.1:11434/api/version
ollama pull qwen3:0.6b                      # один раз
```

Зависимости сверх апстрима не добавлялись: FTS5 входит в `sqlite3`
стандартной библиотеки, pytest нужен только для тестов.

## Свои документы

Приватная база знаний в репозитории **отсутствует намеренно**: `src/docs/`,
`chunks.pkl` (полный текст документов), `index.faiss` (их эмбеддинги) и
`index.fts.sqlite` (их текст в FTS-таблице) закрыты в `.gitignore`.

Положите документы (`.md`, `.txt`, `.pdf`, `.docx`) в `src/docs/`, вложенные
каталоги допустимы:

```bash
cp -r /path/to/your/docs/* src/docs/
git check-ignore -v src/docs/any.md src/chunks.pkl src/index.faiss src/index.fts.sqlite
```

## Индекс

Команды выполняются из `src/`: пути в `config.py` относительные.

```bash
cd src && ../.venv/bin/python main.py build-index
```

Собирает FAISS (`index.faiss`), чанки (`chunks.pkl`) и FTS-индекс
(`index.fts.sqlite`) по одному и тому же списку чанков. Пересобрать только
FTS из сохранённых чанков, не пересчитывая эмбеддинги:

```bash
cd src && ../.venv/bin/python -m rag.fulltext
```

Запуск: `../.venv/bin/python main.py` (интерактивно) или сквозной ответ
одной командой: `../.venv/bin/python -m rag.hybrid "вопрос"`.

Настройки в `src/config.py`: `CHUNK_SIZE=700`, `CHUNK_OVERLAP=100`,
`EMBEDDING_MODEL`, `OLLAMA_MODEL`, `TOP_K=5`, пути индексов. Размер чанка,
перекрытие и модель эмбеддингов в работе не менялись: они задают базу сравнения.

## Замеры

Набор из 15 вопросов с разметкой допустимых файлов: `bench/tasks.py`,
заморожен до первого замера. Метрики: hit@5, MRR, задержка по этапам
(ключевые слова / поиск / слияние), медиана и p95. Из корня репозитория:

```bash
.venv/bin/python -m bench.run --label before --pipeline vector
.venv/bin/python -m bench.run --label after-keywords --pipeline keywords --runs 3
.venv/bin/python -m bench.run --label after-fts --pipeline fts
.venv/bin/python -m bench.run --label after-rrf --pipeline rrf
.venv/bin/python -m bench.run --label after-rrf-weighted --pipeline rrf-weighted
.venv/bin/python -m bench.run --label after-all --pipeline all --runs 3
.venv/bin/python -m bench.run --label after-all-weighted --pipeline all-weighted --runs 3
.venv/bin/python -m bench.report --before before --after after-fts after-all
```

Пайплайны с моделью недетерминированы, для них 3 прогона; в отчёте среднее
и разброс. Отказ на одном вопросе не прерывает остальные, он попадает в
`failed`. Отсутствие индекса или недоступная Ollama дают понятное сообщение.
Результаты лежат в `bench/results/<label>.json`. Каждый замер снят в отдельной
ветке от базы: `feat/keywords`, `feat/fts`, `feat/rrf`, `feat/hybrid`.

Тесты: `.venv/bin/python -m pytest -q` (или `python -m unittest discover -s tests`).

## Почему SQLite FTS5

Спека допускала `rank_bm25` либо SQLite FTS5. Выбран FTS5: он встроен в
стандартную библиотеку Python, ставить ничего не нужно, индекс лежит на диске
рядом с FAISS как производные данные. Ограничение: токенизатор `unicode61`
не знает морфологии русского, «миграций» и «миграции» для него разные слова;
латинские термины (asyncpg, Ruff, slowapi) он берёт надёжно, русские фразы
находит только при совпадении словоформы. Не чинится в этой работе.

## Модель

Ключевые слова и финальный ответ: `qwen3:0.6b` через Ollama, как в апстриме.
Оригинал задания допускает 0.6B–3B; на 0.6B содержание ключевых слов
приемлемое, страдает формат, что лечится терпимым парсером, а не сменой
модели. Известная слабость: на части вопросов модель отвечает служебными
токенами вроде `/no_keywords`, которые проходят парсер как валидные элементы.
Эмбеддинги: `all-MiniLM-L6-v2` из апстрима, англоязычная модель; на русских
вопросах это главная причина слабой базы, см. `REPORT-hybrid.md`.

## Структура

```
src/
├── config.py            настройки
├── main.py              CLI
├── assistant.py         оркестратор RAG + MCP
├── rag/
│   ├── ingest.py        загрузка документов
│   ├── chunk.py         нарезка на чанки
│   ├── embed.py         эмбеддинги
│   ├── build_index.py   FAISS + FTS
│   ├── query.py         векторный поиск, промпт, вызов модели
│   ├── keywords.py      ключевые слова
│   ├── fulltext.py      SQLite FTS5
│   ├── fusion.py        RRF
│   └── hybrid.py        конвейер целиком
├── mcp/                 сервер и клиент MCP
└── docs/                ваши документы (не в git)
bench/                   набор вопросов, метрики, раннер, отчёт, результаты
tests/                   тесты без внешних вызовов
```

## Лицензия

MIT, как в апстриме.
