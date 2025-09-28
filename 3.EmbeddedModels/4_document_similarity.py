from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents=[
    "virat kohli is a good cricket player and captain",
    "ms dhoni is known as capitan cool",
    "rohit sharma won the t20 world cup recently"
]
query='tell me about virat kohli'

doc_embeddings=embedding.embed_documents(documents)
query_embeddings=embedding.embed_query(query)

print(cosine_similarity([query_embeddings],doc_embeddings))