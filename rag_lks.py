#!/usr/bin/env python3
"""
RAG-система для анализа технической документации по линейно-кабельным сооружениям (ЛКС).

Этот модуль реализует Retrieval-Augmented Generation (RAG) пайплайн для
работы с руководством по эксплуатации ЛКС местных сетей связи. Система
работает полностью локально через Ollama и не требует внешних API.

Основные компоненты:
1. Загрузка и разбивка PDF-документа на фрагменты (chunking)
2. Обогащение фрагментов метаданными (номера разделов/пунктов)
3. Векторное представление текста через локальные эмбеддинги
4. Семантический поиск релевантных фрагментов (FAISS)
5. Генерация ответа через локальную LLM

Использование:
    python rag_lks.py

Пример запроса:
    "какие марки муфт применяются для восстановления оболочек кабелей с медными жилами?"
"""

import re
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

PDF_PATH = "docs/rukovod.pdf"
"""Путь к PDF-файлу с технической документацией."""

CHUNK_SIZE = 500
"""Размер фрагмента текста в символах (chunk). Меньше - точнее, но больше фрагментов."""

CHUNK_OVERLAP = 100
"""Перекрытие между соседними фрагментами. Помогает не потерять контекст на границах."""

EMBEDDING_MODEL = "nomic-embed-text"
"""Модель для эмбеддингов (через Ollama). Хорошо понимает русский язык."""

LLM_MODEL = "qwen2.5:1.5b"
"""Языковая модель для генерации ответов. Для лучшего качества используйте 'qwen2.5:7b'."""

LLM_TEMPERATURE = 0.1
"""Температура генерации (0.0-1.0). Низкие значения = более фактологические ответы."""

SEARCH_K = 7
"""Количество фрагментов, извлекаемых из векторной БД для контекста."""

# Ключевые слова для улучшения запроса к пункту 2.1.4
MUFTY_KEYWORDS = ["марки муфт", "марка муфты", "какие муфты"]
SPECIFIC_SECTION = "2.1.4"
SECTION_TAG_TEMPLATE = "пункт раздел {section_num}"


# ============================================================================
# ЗАГРУЗКА И ОБРАБОТКА PDF
# ============================================================================


def load_pdf_document(path: str) -> List[Document]:
    """
    Загружает PDF-документ и возвращает список страниц.

    Args:
        path: Путь к PDF-файлу.

    Returns:
        Список объектов Document, где каждый соответствует одной странице.

    Raises:
        FileNotFoundError: Если файл не найден.
        Exception: При ошибках парсинга PDF.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    print(f"✅ Загружено {len(documents)} страниц")
    return documents


def chunk_documents(
    documents: List[Document], chunk_size: int, overlap: int
) -> List[Document]:
    """
    Разбивает документ на смысловые фрагменты (chunks).

    Разбивка происходит по рекурсивному правилу: сначала пробуем разделить по двойным
    переводам строк, затем по одинарным, затем по пробелам. Это сохраняет целостность
    абзацев и предложений.

    Args:
        documents: Список документов (страниц).
        chunk_size: Максимальный размер фрагмента в символах.
        overlap: Перекрытие между соседними фрагментами.

    Returns:
        Список фрагментов (chunks).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],  # от больших к маленьким
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Получено {len(chunks)} фрагментов")
    return chunks


def extract_section_number(text: str) -> Tuple[bool, str]:
    """
    Извлекает номер раздела/пункта из текста.

    Ищет паттерны типа "2.1.4.", "2.1.4)" или "2.1.4 " в начале фрагмента.

    Args:
        text: Текст фрагмента.

    Returns:
        Кортеж (найден_ли, номер_раздела). Если не найден, возвращает (False, "").
    """
    # Ищем номер пункта: цифры.цифры.цифры, за которыми следует точка, пробел или конец
    patterns = [
        r'^(\d+\.\d+\.\d+)\.',  # в начале строки: "2.1.4."
        r'(\d+\.\d+\.\d+)[\.\s]',  # произвольное место
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return True, match.group(1)
    return False, ""


def enhance_chunks_with_metadata(chunks: List[Document]) -> List[Document]:
    """
    Обогащает фрагменты метаданными (виртуальными тегами) для улучшения поиска.

    Добавляет в начало текста теги вида [раздел X.X.X] [пункт X.X.X].
    Это позволяет поиску лучше находить фрагменты по номерам разделов,
    которые плохо различимы в обычных эмбеддингах.

    Args:
        chunks: Список исходных фрагментов.

    Returns:
        Список фрагментов с добавленными тегами.
    """
    enhanced_chunks = []

    for chunk in chunks:
        content = chunk.page_content
        found, section_num = extract_section_number(content)

        if found:
            # Добавляем теги в начало контента для улучшения поисковой индексации
            tags = SECTION_TAG_TEMPLATE.format(section_num=section_num)
            enhanced_content = f"[{tags}] {content}"
            chunk.page_content = enhanced_content

    print(
        f"✅ Обогащено {len([c for c in chunks if c.page_content != c.page_content])} фрагментов"
    )
    return chunks


# ============================================================================
# ВЕКТОРНАЯ БАЗА ДАННЫХ
# ============================================================================


def create_vector_store(chunks: List[Document], embedding_model: str) -> FAISS:
    """
    Создаёт векторную базу данных FAISS из фрагментов.

    Превращает текст в векторные представления (эмбеддинги) и строит индекс
    для быстрого семантического поиска.

    Args:
        chunks: Список фрагментов с текстом.
        embedding_model: Название модели эмбеддингов в Ollama.

    Returns:
        FAISS векторная база данных.
    """
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_db = FAISS.from_documents(chunks, embeddings)
    print("✅ Векторная база (FAISS) готова")
    return vector_db


# ============================================================================
#  ПОИСК КОНТЕКСТА
# ============================================================================


def enhance_query_for_section(query: str) -> str:
    """
    Улучшает запрос, добавляя ключевые слова для поиска конкретных разделов.

    Если запрос содержит специфические термины (например, "марки муфт"),
    добавляется подсказка для поиска по номеру раздела. Это повышает
    точность поиска для документов с чёткой нумерацией.

    Args:
        query: Исходный запрос пользователя.

    Returns:
        Улучшенный запрос.
    """
    query_lower = query.lower()
    for keyword in MUFTY_KEYWORDS:
        if keyword in query_lower:
            # Добавляем подсказку о номере раздела
            section_hint = SECTION_TAG_TEMPLATE.format(
                section_num=SPECIFIC_SECTION
            )
            return f"{query} {section_hint}"
    return query


def retrieve_context(query: str, vector_db: FAISS, k: int = SEARCH_K) -> str:
    """
    Извлекает релевантный контекст из векторной базы данных.

    Выполняет семантический поиск по эмбеддингам, возвращает k наиболее
    похожих фрагментов, объединённых в строку.

    Args:
        query: Запрос пользователя.
        vector_db: Векторная база FAISS.
        k: Количество извлекаемых фрагментов.

    Returns:
        Строка с объединёнными фрагментами, разделёнными ---.
    """
    # Улучшаем запрос для более точного поиска
    enhanced_query = enhance_query_for_section(query)

    # Поиск по семантической близости
    docs = vector_db.similarity_search(enhanced_query, k=k)

    # Объединяем с разделителем для сохранения границ между фрагментами
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return context


# ============================================================================
# LLM И RAG ЦЕПОЧКА
# ============================================================================


def create_llm(model: str, temperature: float) -> ChatOllama:
    """
    Создаёт клиент для локальной LLM через Ollama.

    Args:
        model: Название модели (например, "qwen2.5:1.5b").
        temperature: Температура генерации (0.0-1.0).

    Returns:
        Настроенный объект ChatOllama.
    """
    llm = ChatOllama(model=model, temperature=temperature)
    print(f"✅ LLM загружена: {model}")
    return llm


def create_rag_prompt() -> ChatPromptTemplate:
    """
    Создаёт шаблон промпта для RAG-цепочки.

    Промпт требует от LLM отвечать строго по контексту и не выдумывать
    информацию. Это ключевое требование для борьбы с галлюцинациями.

    Returns:
        Шаблон промпта.
    """
    template = """
Ты — помощник для анализа технической документации по линейно-кабельным сооружениям.
Отвечай на вопрос, используя ТОЛЬКО контекст из документа.

ПРАВИЛА:
1. Если ответ есть в контексте — ответь кратко и по делу.
2. Если ответа нет в контексте — скажи "В документе это не описано".
3. НЕ выдумывай и НЕ добавляй информацию из своего знания.
4. В ответе можно цитировать номера пунктов из контекста.

Контекст:
{context}

Вопрос пользователя: {question}

Ответ:
"""
    return ChatPromptTemplate.from_template(template)


def setup_rag_chain(vector_db: FAISS, llm: ChatOllama):
    """
    Настраивает полную RAG-цепочку (Retrieval -> Augmentation -> Generation).

    Использует LangChain Expression Language (LCEL) для композиции компонентов:
    1. Извлечение контекста по запросу
    2. Формирование промпта
    3. Генерация ответа LLM
    4. Парсинг вывода

    Args:
        vector_db: Векторная база для поиска.
        llm: Языковая модель для генерации.

    Returns:
        Callable RAG-цепочка.
    """
    prompt = create_rag_prompt()

    # LCEL цепочка: context из vector_db, затем prompt, llm, парсер
    rag_chain = (
        {
            "context": lambda x: retrieve_context(x["question"], vector_db),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================


def test_search(vector_db: FAISS) -> None:
    """
    Выполняет тестовый поиск для проверки качества индексации.

    Ищет пункт 2.1.4 (с марками муфт) и сообщает, был ли он найден.

    Args:
        vector_db: Векторная база.
    """
    test_query = (
        "какие марки муфт для восстановления оболочек кабелей с медными жилами"
    )
    print(f"\n🔍 Проверка поиска: '{test_query[:50]}...'")

    context = retrieve_context(test_query, vector_db)

    if SPECIFIC_SECTION in context:
        print(f"✅ Поиск нашёл пункт {SPECIFIC_SECTION}!")
    else:
        print(f"❌ Поиск не нашёл {SPECIFIC_SECTION}, но вот что нашёл:")
        print(context[:500])


# ============================================================================
# ИНТЕРАКТИВНЫЙ ЧАТ
# ============================================================================


def run_interactive_chat(rag_chain, vector_db: FAISS) -> None:
    """
    Запускает интерактивный цикл для общения с RAG-ботом.

    Args:
        rag_chain: RAG-цепочка для обработки запросов.
        vector_db: Векторная база (используется только для теста).
    """
    # Сначала тест
    test_search(vector_db)

    print("\n" + "=" * 50)
    print("💬 RAG-бот готов! Задавай вопросы по документу ЛКС.")
    print("=" * 50 + "\n")

    while True:
        try:
            question = input("❓ Вопрос (exit для выхода): ")
        except (KeyboardInterrupt, EOFError):
            print("\nВыход...")
            break

        if question.lower() in ["exit", "выход", "quit", "q"]:
            print("До свидания!")
            break

        if not question.strip():
            print("Пожалуйста, введите вопрос.")
            continue

        print("🤔 Думаю...")
        try:
            answer = rag_chain.invoke({"question": question})
            print(f"\n📢 Ответ:\n{answer}\n")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print(
                "Убедитесь, что Ollama запущен (ollama serve) и модели скачаны."
            )


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Основная функция: загружает документ, строит индекс и запускает чат."""
    print("=" * 60)
    print("RAG-система для анализа документации по ЛКС")
    print("=" * 60 + "\n")

    # 1. Загрузка PDF
    documents = load_pdf_document(PDF_PATH)

    # 2. Разбивка на фрагменты
    raw_chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3. Обогащение метаданными
    enhanced_chunks = enhance_chunks_with_metadata(raw_chunks)

    # 4. Создание векторной базы
    vector_db = create_vector_store(enhanced_chunks, EMBEDDING_MODEL)

    # 5. Инициализация LLM
    llm = create_llm(LLM_MODEL, LLM_TEMPERATURE)

    # 6. Сборка RAG-цепочки
    rag_chain = setup_rag_chain(vector_db, llm)

    # 7. Запуск интерактивного чата
    run_interactive_chat(rag_chain, vector_db)


if __name__ == "__main__":
    main()
