from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Load pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Free model on HF
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100
    }
)

# Wrap pipeline into ChatModel
model = ChatHuggingFace(llm=llm)

# Invoke
res = model.invoke("What is the capital of India?")
print(res.content)
