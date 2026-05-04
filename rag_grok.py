from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

GROQ_API_KEY = "gsk_qhRVdJRqLj12K7I5CvJOWGdyb3FYPvNsUFyHLItGYY0Kpyw8YO**"

print("📄 Загружаю PDF...")
loader = PyPDFLoader("docs/rukovod.pdf")
documents = loader.load()
print(f"✅ Загружено {len(documents)} страниц")

print("✂️ Режу документ на куски...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Получилось {len(chunks)} кусков")

print("🧠 Создаю эмбеддинги...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)
print("✅ Векторная база готова")

print("🤖 Подключаю Groq...")
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")


# Создаём функцию поиска
def get_context(query):
    results = []
    query_lower = query.lower()

    # Добавляем результаты векторного поиска
    vector_results = vector_db.similarity_search(query, k=5)
    for doc in vector_results:
        if doc.page_content not in results:
            results.append(doc.page_content)

    return "\n\n---\n\n".join(results[:5])


# Промпт
prompt = ChatPromptTemplate.from_template("""
Ты помощник, который отвечает на вопросы строго по документу.

Контекст из документа:
{context}

Вопрос: {question}

Если ответа нет в контексте, скажи "В документе это не описано".
Ответь кратко и по существу:
""")

# Создаём цепочку
chain = (
    {
        "context": lambda x: get_context(x["question"]),
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "=" * 50)
print("💬 Готово! Задавай вопросы по документу:")
print("=" * 50 + "\n")

# Тестовый поиск
test_question = "какие марки муфт применяются для восстановления оболочек кабелей с медными жилами?"
print(f"🔍 Тестовый поиск: {test_question[:50]}...")
test_context = get_context(test_question)
if "2.1.4" in test_context:
    print("✅ Найден пункт 2.1.4 с марками муфт!\n")

while True:
    question = input("❓ Вопрос (выход = exit): ")
    if question.lower() in ["exit", "выход", "quit"]:
        break

    print("🤔 Думаю...")
    answer = chain.invoke({"question": question})
    print(f"\n📢 Ответ:\n{answer}\n")
    print("-" * 50)
