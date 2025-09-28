from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1️⃣ Prompts
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text:\n{text}',
    input_variables=['text']
)

# 2️⃣ Load small, fast model
llm = HuggingFacePipeline.from_model_id(
    model_id="distilgpt2",   # small & fast
    task="text-generation" # limit output for speed
)

# 3️⃣ Output parser
parser = StrOutputParser()

# 4️⃣ Chain: report -> summary
chain = prompt1 | llm | parser | prompt2 | llm | parser

# 5️⃣ Run
res = chain.invoke({'topic': 'Unemployment in India'})
print(res)
