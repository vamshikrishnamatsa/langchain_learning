from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Hugging Face endpoint setup
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",  # good free model
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

res = model.invoke("What is the capital of India?")
print(res.content)

