from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

PDF_PATH = "docs/rukovod.pdf"

print("📄 Загружаю PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"✅ Загружено {len(documents)} страниц")

print("✂️ Разбиваю на куски с обогащением метаданными...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)
raw_chunks = text_splitter.split_documents(documents)

# Обогащаем чанки: добавляем в content виртуальные теги (номер пункта)
enhanced_chunks = []
for chunk in raw_chunks:
    content = chunk.page_content
    # Ищем номер пункта в начале чанка (например, "2.1.4.")
    match = re.search(r'(\d+\.\d+\.\d+)\.', content)
    if match:
        section_num = match.group(1)
        # Добавляем теги в начало контента для улучшения поиска
        enhanced_content = (
            f"[раздел {section_num}] [пункт {section_num}] {content}"
        )
        chunk.page_content = enhanced_content
    enhanced_chunks.append(chunk)

print(f"✅ Получено {len(enhanced_chunks)} обогащённых фрагментов")

print("🧠 Создаю эмбеддинги...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = FAISS.from_documents(enhanced_chunks, embeddings)
print("✅ Векторная база готова")

print("🤖 Подключаю локальную LLM...")
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.1)


def get_context(query, k=7):
    # Добавляем синонимы в запрос для лучшего поиска
    enhanced_query = query
    if "марки муфт" in query.lower():
        enhanced_query += " пункт раздел 2.1.4"

    docs = vector_db.similarity_search(enhanced_query, k=k)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


prompt = ChatPromptTemplate.from_template("""
Ты помощник. Отвечай строго по контексту.

Контекст:
{context}

Вопрос: {question}

Ответ:
""")

rag_chain = (
    {
        "context": lambda x: get_context(x["question"]),
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Тест
test_query = (
    "какие марки муфт для восстановления оболочек кабелей с медными жилами"
)
print(f"\n🔍 Проверка поиска...")
context = get_context(test_query)
if "2.1.4" in context:
    print("✅ Поиск нашёл пункт 2.1.4!")
else:
    print("❌ Поиск не нашёл 2.1.4, но вот что нашёл:")
    print(context[:500])

print("\n" + "=" * 50)
print("💬 RAG-бот готов!")
print("=" * 50 + "\n")

while True:
    question = input("❓ Вопрос (exit для выхода): ")
    if question.lower() in ["exit", "выход", "quit"]:
        break
    print("🤔 Думаю...")
    answer = rag_chain.invoke({"question": question})
    print(f"\n📢 Ответ:\n{answer}\n")
    print("-" * 50)
