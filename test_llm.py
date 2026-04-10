# test_llm.py in backend/
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL")
)
response = llm.invoke("What is RAG in one sentence?")
print(response.content)