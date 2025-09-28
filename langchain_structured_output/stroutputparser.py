from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# prompt-1 detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# prompt-2 summary
template2 = PromptTemplate(
    template="Write a one-line summary of the following text:\n{text}",
    input_variables=["text"]
)

# Generate detailed report
prompt1 = template1.format(topic="black hole")
result = model.invoke(prompt1)

# Generate summary from the detailed report
prompt2 = template2.format(text=result.content)
result1 = model.invoke(prompt2)

print("Detailed Report:\n", result.content)
print("\nOne-line Summary:\n", result1.content)
