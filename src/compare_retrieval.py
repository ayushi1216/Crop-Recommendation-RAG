from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Load row-level index
row_db = FAISS.load_local(
    "faiss_index", embedding_model, allow_dangerous_deserialization=True
)

# Load aggregate index
agg_db = FAISS.load_local(
    "faiss_aggregate_index", embedding_model, allow_dangerous_deserialization=True
)

query = "Compare rice and maize nitrogen requirements"

print("\n==============================")
print("ROW-LEVEL RETRIEVAL")
print("==============================")

row_results = row_db.similarity_search(query, k=3)

for idx, r in enumerate(row_results, 1):

    print(f"\nResult {idx}")
    print(r.page_content)

print("\n==============================")
print("AGGREGATE RETRIEVAL")
print("==============================")

agg_results = agg_db.similarity_search(query, k=3)

for idx, r in enumerate(agg_results, 1):

    print(f"\nResult {idx}")
    print(r.page_content)
