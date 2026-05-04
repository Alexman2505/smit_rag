# search_terms.py
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/rukovod.pdf")
documents = loader.load()

text = " ".join([doc.page_content for doc in documents])

# Ищем упоминания
search_terms = ["муфт", "марк", "кабель", "восстановл"]

for term in search_terms:
    count = text.lower().count(term.lower())
    print(f"'{term}': найдено {count} раз(а)")
