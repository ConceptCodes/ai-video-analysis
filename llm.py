from langchain_ollama import ChatOllama

vision_llm = ChatOllama(model="llava-llama3", temperature=0)
base_llm = ChatOllama(model="llama3.1", temperature=0.3)
# base_llm = ChatOllama(model="deepseek-r1", temperature=0.3)
