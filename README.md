# Local RAG/MCP Knowledge Base Assistant

# 🛠️ Окружение с нуля (форк, задание 5)

Приватная база знаний в репозитории **отсутствует намеренно**: `src/docs/`,
`chunks.pkl` (полный текст документов) и `index.faiss` (их эмбеддинги) закрыты
в `.gitignore`. Положите свои документы (`.md`, `.txt`, `.pdf`, `.docx`)
в `src/docs/`, вложенные каталоги допустимы.

Требования: Python 3.10+, запущенная Ollama с моделью `qwen3:0.6b`.

```bash
# 1. Виртуальное окружение в корне проекта и зависимости апстрима
python3 -m venv .venv                       # если нет ensurepip: uv venv .venv --seed
.venv/bin/pip install -r src/requirements.txt   # или: uv pip install --python .venv/bin/python -r src/requirements.txt

# 2. Ollama: проверить, что сервер отвечает и модель на месте
curl -s http://127.0.0.1:11434/api/version
ollama pull qwen3:0.6b                      # один раз

# 3. Документы и индекс (команды выполняются из src/, пути в config.py относительные)
cp -r /path/to/your/docs/* src/docs/
cd src && ../.venv/bin/python main.py build-index

# 4. Интерактивный режим
../.venv/bin/python main.py
```

Проверка, что документы и производные данные не попадут в git:

```bash
git check-ignore -v src/docs/any.md src/chunks.pkl src/index.faiss
```

# 🔎 Гибридный поиск (задание 5): принятые решения

- **Query Expansion: ключевые слова, а не альтернативные формулировки.**
  `src/rag/keywords.py`, один вызов `qwen3:0.6b` через Ollama, `temperature=0`,
  `think: false` (без этого блок рассуждений на CPU не укладывается в минуты).
  `num_thread` в запрос не передаётся: значение зашито в модель, своё перебило бы его.
- **В поиск уходит исходный вопрос плюс ключевые слова** одной строкой
  (`expand_query`): вопрос, перевод строки, ключевые слова через запятую.
- **Парсер ответа терпимый**: режет по запятым и переводам строк, снимает
  маркеры списков и эхо метки `Keywords:`, убирает пустые и дубликаты (без учёта
  регистра), выбрасывает элементы длиннее 60 символов. Fallback на исходный
  вопрос только если ничего не разобралось или ответ длиннее 200 символов;
  в лог уходит предупреждение, прогон продолжается.
- **Бенчмарк**: `python -m bench.run --label <label> --pipeline <vector|keywords> --runs N`,
  `python -m bench.report --before before --after <labels...>`. Шаг с моделью
  недетерминирован, для него 3 прогона. Итоги и предсказания: `REPORT-hybrid.md`.
- **Тесты**: `pytest -q` (спека) или без установки pytest
  `.venv/bin/python -m unittest discover -s tests`; классы `unittest.TestCase`
  собираются обоими раннерами. Внешних вызовов в тестах нет.
# 🔎 Гибридный поиск (задание 5): полнотекстовый поиск

- **Выбор: SQLite FTS5**, не `rank_bm25`. Он встроен в стандартную библиотеку
  Python (`sqlite3`), ставить ничего не нужно, индекс лежит на диске рядом с
  FAISS (`src/index.fts.sqlite`, производные данные под `.gitignore`).
- `src/rag/fulltext.py`: `build_fts(chunks)` строит таблицу по тем же чанкам,
  что и FAISS, `rowid` = позиция чанка в списке = идентификатор, который
  возвращает FAISS. `search_fts(query, top_k)` возвращает `[(chunk_id, rank)]`,
  ранг с единицы, порядок по `bm25()`.
- Запрос превращается в выражение MATCH: каждое слово в кавычках, между
  словами `OR`. Так пунктуация и `--force-with-lease` не ломают синтаксис,
  а bm25 всё равно поднимает чанки с более редкими словами.
- Индекс строится в `build_index` вместе с FAISS. Пересобрать только FTS
  из сохранённого `chunks.pkl`, не пересчитывая эмбеддинги:
  `cd src && ../.venv/bin/python -m rag.fulltext`.
- **Ограничение**: токенизатор `unicode61` не знает морфологии русского,
  «миграций» и «миграции» для него разные слова. Подробности в `REPORT-hybrid.md`.
- Бенчмарк: `python -m bench.run --label after-fts --pipeline fts` (только FTS,
  без вектора и ключевых слов). Тесты: `pytest -q` (pytest установлен в `.venv`).

# 🔎 Гибридный поиск (задание 5): слияние RRF

- `src/rag/fusion.py`: `reciprocal_rank_fusion(runs, k=60, weights=None)` принимает
  список ранжированных списков идентификаторов и возвращает `[(chunk_id, score)]`,
  лучший первым. `score(d) = Σ w_i / (k + rank_i(d))`, ранг с единицы, документ из
  одного списка получает одно слагаемое. Ничьи решаются порядком первого
  появления, результат детерминирован. О поиске модуль не знает.
- `search_vector(query, top_k)` в `rag/query.py` — новая функция, отдающая
  `[(chunk_id, rank)]` для слияния; `retrieve` не менялась.
- **Глубина кандидатов**: каждый поиск отдаёт 20 кандидатов (`FUSION_CANDIDATES`
  в `bench/run.py`), слияние режется до топ-5. Выбрано до замера, не подбиралось.
- Пайплайны бенчмарка: `rrf` (веса 1.0/1.0) и `rrf-weighted` (вектор 0.3,
  FTS 1.0, проверка гипотезы о неравных компонентах). В этом шаге поиски
  выполняются последовательно, параллельный запуск — следующий шаг.

# 🔎 Гибридный поиск (задание 5): конвейер целиком

- `src/rag/hybrid.py`: `hybrid_search(question)` — ключевые слова → векторный
  поиск и FTS **параллельно** в `ThreadPoolExecutor` (два блокирующих поиска,
  пул открывается на вызов и закрывается контекстным менеджером) → RRF → топ-5.
  `retrieve_hybrid` отдаёт чанки для промпта, `ask_hybrid` доводит до ответа
  модели. `assistant.py` теперь берёт контекст из `retrieve_hybrid`; `main.py`
  без изменений. В `ask_llm` добавлен `think: false`.
- Отказ FTS не роняет запрос: предупреждение в лог, выдача из одного
  векторного списка. Отказ вектора поднимается наверх — искать больше нечем.
- Веса RRF — параметр `weights`, по умолчанию `(1.0, 1.0)`. Значение 0.3 для
  вектора проверено отдельным замером и не подбиралось.
- Задержки считаются раздельно: ключевые слова, каждый поиск по отдельности,
  фаза параллельного поиска (стенка), слияние. Бенчмарк: `--pipeline all`
  и `--pipeline all-weighted`, по 3 прогона, в цепочке есть модель.
- Пример сквозного ответа: `cd src && ../.venv/bin/python -m rag.hybrid "вопрос"`.

# 📋 The Problem

- **Growing Documentation**: Knowledge scattered across files
- **Information Retrieval**: Hard to find answers without keywords
- **Privacy Concerns**: Cloud solutions may not comply with policies

```
Users → Search → Answer = 😫
```

# ✨ The Solution

A **local, intelligent Q&A system** using:

- **RAG**: Semantic search over documentation
- **MCP**: Dynamic document access
- **Local LLM**: Privacy-preserving answers (Ollama)

# ✨ Key Benefits

- ✅ Privacy-first (runs locally)
- ✅ No API costs
- ✅ Fast semantic search
- ✅ Intelligent document access
- ✅ Complete data control

# 🏗️ Architecture - Top Level

```
┌──────────────────────┐
│   User Interface     │ (CLI)
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  [RAG]       [MCP]
   Query      Tools
     │           │
     └─────┬─────┘
           ▼
    [Ollama LLM]
```

# 🏗️ Architecture - Storage

```
┌────────────────┐
│  FAISS Index   │ Vector Database
│  + MCP Tools   │
└────────┬───────┘
         │
    ┌────▼─────┐
    │   docs/  │
    │directory │
    └──────────┘
```

# 🔍 RAG Pipeline

1. Document Loading → Read .md, .txt, .pdf, .docx
2. Chunking → Split into 700-char chunks
3. Embedding → Use SentenceTransformers
4. Indexing → Build FAISS vector index
5. Query → Retrieve top 5 similar chunks
6. Prompt Building → Create context-aware prompt
7. LLM Generation → Get answer from model

# 🔍 Why FAISS?

- Fast vector similarity search
- Lightweight and memory-efficient
- No external dependencies
- Perfect for local deployments
- Millions of vectors supported

# 🔧 MCP - Model Context Protocol

MCP provides **standardized interface** for LLM tool access:

```python
read_document(file_path)
list_documents()
search_documents(query)
```

# 🔧 MCP Benefits

- Tool Use by LLM
- Real-time document access
- Standardized interface
- Easy to extend
- Local tool execution

# 💻 Tech Stack

```
Language:      Python 3.10+
Vector DB:     FAISS
Embeddings:    SentenceTransformers
LLM:           Ollama (local)
MCP:           FastMCP
```

# 📁 Project Structure

```
src/
├── config.py           Configuration
├── main.py             CLI entry point
├── assistant.py        Main orchestrator
├── rag/
│   ├── ingest.py      Load documents
│   ├── chunk.py       Split text
│   ├── embed.py       Generate embeddings
│   ├── build_index.py Build FAISS index
│   └── query.py       Retrieve & generate
├── mcp/
│   ├── server.py      MCP tool definitions
│   └── client.py      MCP client wrapper
└── docs/              Documentation
```

# 🚀 Index Building (Setup)

```
$ python main.py build-index

1. Load documents
  ↓
2. Split into chunks
  ↓
3. Generate embeddings
  ↓
4. Build FAISS index
  ↓
5. Save files
```

# 🚀 Query Processing (Runtime)

```
User Question
  ↓
Embed question
  ↓
Search FAISS → Top 5 chunks
  ↓
LLM decides: Use MCP tools?
  ↓
Build prompt + context
  ↓
Call Ollama
  ↓
Return answer + sources
```

# ✨ Core Features

- **Semantic Search**: Find by meaning, not keywords
- **Multi-format**: .md, .txt, .pdf, .docx files
- **Source Attribution**: Shows document sources
- **MCP Tools**: LLM can read full documents
- **No External APIs**: Runs locally only
- **Fast Retrieval**: Sub-second search

# ⚙️ Configuration Options

```python
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:0.6b"
TOP_K = 5
```

# 🎬 Live Demo - Starting

```bash
$ python main.py
```

Output:
```
🤖 Company Knowledge Base
Ask questions about documentation
Type 'exit' to stop
```

# 🎬 Demo - Query 1

```
❓ What are company values?

🤖 Innovation, integrity, collaboration

📚 Sources:
  • Loan Rangers Team.md
  • Info Security.md
```

# 🎬 Demo - Query 2

```
❓ What documents do we have?

🤖 [Uses MCP list_documents]
  • Loan Rangers Team.md
  • Information Security.md
  • Services.md
```

# 🎬 Demo - Query 3

```
❓ Full security policy?

🤖 [Uses MCP read_document]
[Full document content...]
```

# 🔐 Security - Local vs Cloud

**Cloud**: Data → Internet → Server
- ⚠️ Network transmission
- ⚠️ External storage
- ⚠️ Subscription costs

**Local**: Data → Local System
- ✅ No transmission
- ✅ Local storage only
- ✅ No costs

# 🔐 Implementation Safeguards

- **MCP Sandbox**: Prevents path traversal
- **Local Storage**: Documents stay on device
- **No Telemetry**: No tracking
- **Offline Ready**: Works without internet

# ⚡ Performance Benchmarks

```
Index Building:   ~30s (one-time)
Query Embedding:  ~50ms
FAISS Search:     ~5ms
LLM Generation:   2-5s
Total Cycle:      2-6s
```

# ⚡ Tuning for Speed

```python
# Faster (smaller model):
OLLAMA_MODEL = "qwen3:0.6b"

# Faster retrieval:
TOP_K = 3
CHUNK_SIZE = 500
```

# 🚢 Deployment - Single Machine

```
1. Install Ollama & Python deps
2. Copy docs/ to server
3. Build index
4. Run with nohup

$ nohup python main.py > log &
```

# 🚢 Scaling - Option 1: FastAPI

```
[HTTP Clients]			[HTTP Clients + Webllm]
       ↓        						 ↓
   [FastAPI]     				 [FastAPI]
       ↓         					 ↓
[Ollama + FAISS]      			  [FAISS]
```

# 🚢 Scaling - Option 2: Distributed

```
[Clients] → [Load Balancer]
             ↓
      [Multiple Retrievers]
```

# 🚢 Storage Scaling

```
Docs     Index      Build
10 MB    ~2 MB      ~5s
100 MB   ~20 MB     ~30s
1 GB     ~200 MB    ~5min
```

# 🔮 Phase 2: Enhanced Features

- ☐ Web UI (Streamlit)
- ☐ API endpoints
- ☐ Multi-language support
- ☐ Document versioning
- ☐ Fine-tuned embeddings

# 🔮 Phase 3: Advanced

- ☐ Conversation memory
- ☐ Multi-hop reasoning
- ☐ Metadata filtering
- ☐ Feedback loop
- ☐ Analytics dashboard

# 🔮 Phase 4: Enterprise

- ☐ User authentication
- ☐ Audit logging
- ☐ Role-based access
- ☐ LLM fine-tuning
- ☐ Cost analysis

# 📊 Why This Works

| Aspect | Traditional | Our RAG |
|--------|---|---|
| **Understanding** | Keywords | Semantic |
| **Answers** | Documents | Direct |
| **Privacy** | Cloud | Local |
| **Cost** | Subscription | One-time |
| **Speed** | Slow | Sub-second |

# ✅ What You Have Now

- Local privacy-first knowledge base
- Fast semantic search (FAISS)
- Intelligent tool use (MCP)
- Maintainable Python code
- Foundation for enterprise features

# 🙋 Quick Reference

```bash
# Build index
python main.py build-index

# Run interactively
python main.py

# Check config
cat config.py
```

# 📚 Resources

- **Code**: MobilaName/local-rag-mcp
- **FAISS**: facebook/faiss
- **Ollama**: ollama.ai
- **FastMCP**: github.com/jlowin/fastmcp
- **Transformers**: huggingface.co

**Thank You!**
