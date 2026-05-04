import os
from groq import Groq

GROQ_API_KEY = "gsk_JkJBaO0mzVrYejy1vgYvWGdyb3FYEHDj**************"

client = Groq(api_key=GROQ_API_KEY)

try:
    # Отправляем запрос на получение списка моделей
    models = client.models.list()

    print("✅ Доступные вам модели Groq:")
    print("-" * 30)
    for model in models.data:
        print(f"• {model.id}")

except Exception as e:
    print(f"❌ Ошибка при получении списка моделей: {e}")
