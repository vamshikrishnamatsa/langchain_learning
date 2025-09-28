from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()

model1=ChatOpenAI()
model2=ChatAnthropic(model_name='claude-3')

prompt1=PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge the provided notes and quiz into a singel document \n notes ->{notes} and {quiz}',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser

}
)

merge_chain=prompt3 | model1 | parser 

chain=parallel_chain | merge_chain
text="""
Virat Kohli is an Indian cricketer widely regarded as one of the best batsmen in the world. Born on November 5, 1988, in Delhi, he made his debut for the Indian national team in 2008. Kohli is known for his aggressive batting style, consistency, and remarkable fitness. He has served as the captain of the Indian cricket team across all formats and has numerous records to his name, including fastest centuries in ODIs and being the fastest player to reach 8,000, 9,000, 10,000, and 11,000 runs in One Day Internationals. Off the field, Kohli is also known for his philanthropic work, endorsements, and his passion for fitness and healthy living.
"""
res=chain.invoke({'text':text})

print(res)

