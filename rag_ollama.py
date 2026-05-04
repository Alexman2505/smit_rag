from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "docs/rukovod.pdf"

print("📄 Загружаю PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"✅ Загружено {len(documents)} страниц")

print("✂️ Разбиваю на куски...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Получено {len(chunks)} фрагментов")

print("🧠 Создаю эмбеддинги через Ollama...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = FAISS.from_documents(chunks, embeddings)
print("✅ Векторная база готова")

print("🤖 Подключаю локальную LLM...")
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
Отвечай на вопрос, используя ТОЛЬКО контекст. Если ответа нет — скажи "В документе не описано".

Контекст: {context}
Вопрос: {question}
Ответ:
""")


def format_docs(docs):
    return "\n\n---\n\n".join([d.page_content for d in docs])


rag_chain = (
    {
        "context": lambda x: format_docs(
            vector_db.similarity_search(x["question"], k=5)
        ),
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "=" * 50)
print("💬 RAG-бот готов! Задавай вопросы по документу:")
print("=" * 50 + "\n")

while True:
    question = input("❓ Вопрос (exit для выхода): ")
    if question.lower() in ["exit", "выход", "quit"]:
        break
    print("🤔 Думаю...")
    answer = rag_chain.invoke({"question": question})
    print(f"\n📢 Ответ:\n{answer}\n")
    print("-" * 50)
