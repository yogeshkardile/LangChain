from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-6-luna")

response = llm.invoke("What is machine learning?")
print(response)
