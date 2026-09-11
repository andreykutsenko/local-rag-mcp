"""Frozen benchmark question set for assignment 5 (hybrid search).

Fixed BEFORE the first measurement; must not change between runs.
Answer paths are relative to src/docs. A question counts as answered
if ANY of its answer files appears in the retrieved top-K.
"""

EXACT = "exact"
SEMANTIC = "semantic"
MIXED = "mixed"

BACKEND_DB = "Документация/Бэкенд/База данных"
BACKEND = "Документация/Бэкенд"
FRONTEND = "Документация/Фронтенд"
FRONTEND_SETUP = f"{FRONTEND}/02-Настройка и конфигурация"

TASKS = [
    # --- 6 exact terms: FTS is expected to win ---
    {
        "id": 1,
        "kind": EXACT,
        "question": "Какие настройки нужны для asyncpg?",
        "answers": [f"{BACKEND_DB}/Использование asyncpg.md"],
    },
    {
        "id": 2,
        "kind": EXACT,
        "question": "Какая длина строки задана для Ruff?",
        "answers": [f"{BACKEND}/Инструменты качества кода (Ruff, Pre-commit).md"],
    },
    {
        "id": 3,
        "kind": EXACT,
        "question": "Какие хуки запускает pre-commit?",
        "answers": [
            f"{BACKEND}/Инструменты качества кода (Ruff, Pre-commit).md",
            f"{FRONTEND_SETUP}/Стиль кода.md",
        ],
    },
    {
        "id": 4,
        "kind": EXACT,
        "question": "Что такое FSD и какие слои в нём есть?",
        "answers": [
            f"{FRONTEND}/01-Обзор проекта/Архитектура.md",
            f"{FRONTEND}/01-Обзор проекта/О проекте.md",
            f"{FRONTEND}/README.md",
        ],
    },
    {
        "id": 5,
        "kind": EXACT,
        "question": "Как инициализировать slowapi в FastAPI?",
        "answers": [f"{BACKEND}/Ограничение частоты запросов (Rate Limiting).md"],
    },
    {
        "id": 6,
        "kind": EXACT,
        "question": "Когда использовать git push --force-with-lease?",
        "answers": ["Документация/Работа с Git.md"],
    },
    # --- 4 semantic questions: vector search is expected to win ---
    {
        "id": 7,
        "kind": SEMANTIC,
        "question": "Что произойдёт, если клиент отправит слишком много запросов подряд?",
        "answers": [f"{BACKEND}/Ограничение частоты запросов (Rate Limiting).md"],
    },
    {
        "id": 8,
        "kind": SEMANTIC,
        "question": "Как накатить изменения в базе данных?",
        "answers": [
            f"{BACKEND_DB}/Миграции.md",
            f"{BACKEND_DB}/migration_policy.md",
            f"{BACKEND_DB}/Инициализация БД.md",
        ],
    },
    {
        "id": 9,
        "kind": SEMANTIC,
        "question": "Почему браузер ругается на сертификат при локальном запуске?",
        "answers": ["Документация/Локальный запуск/Прокси.md"],
    },
    {
        "id": 10,
        "kind": SEMANTIC,
        "question": "Кто в команде отвечает за настройку CI и деплой?",
        "answers": ["Роли/DevOps | CI-CD Engineer.md"],
    },
    # --- 5 mixed questions: hybrid is expected to win ---
    {
        "id": 11,
        "kind": MIXED,
        "question": "Какие переменные окружения фронтенда реально используются?",
        "answers": [
            f"{FRONTEND_SETUP}/Переменные окружения.md",
            f"{FRONTEND_SETUP}/Docker.md",
            f"{FRONTEND_SETUP}/Установка и запуск.md",
        ],
    },
    {
        "id": 12,
        "kind": MIXED,
        "question": "Какой формат ошибок возвращает API?",
        "answers": [
            f"{BACKEND}/Формат ошибок.md",
            f"{BACKEND}/spaces_endpoints.md",
            f"{BACKEND_DB}/Архитектура бэкенда.md",
            f"{FRONTEND}/05-Разработка/Решение проблем.md",
            f"{FRONTEND}/04-Функциональность/Авторизация.md",
        ],
    },
    {
        "id": 13,
        "kind": MIXED,
        "question": "Какая политика миграций принята в проекте?",
        "answers": [
            f"{BACKEND_DB}/migration_policy.md",
            f"{BACKEND_DB}/Миграции.md",
            f"{BACKEND_DB}/Архитектура бэкенда.md",
        ],
    },
    {
        "id": 14,
        "kind": MIXED,
        "question": "Где хранятся секреты для CI и где их настраивать в GitHub?",
        "answers": [
            f"{BACKEND_DB}/Переменные окружения.md",
            "Календари/Сроки действия.md",
        ],
    },
    {
        "id": 15,
        "kind": MIXED,
        "question": "На каких портах доступны frontend, API и pgAdmin при локальном запуске?",
        "answers": [
            "Документация/Локальный запуск/Прокси.md",
            f"{FRONTEND_SETUP}/Docker.md",
            f"{FRONTEND_SETUP}/Установка и запуск.md",
            f"{BACKEND_DB}/Инициализация БД.md",
            f"{BACKEND_DB}/Переменные окружения.md",
        ],
    },
]
