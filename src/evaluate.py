import json

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# LOAD EVALUATION DATASET
# ==========================================================
# Evaluation set contains:
# - natural language question
# - expected crop label
#
# Purpose:
# Evaluate retrieval quality of the RAG system.

with open("evaluation/evaluation_set.json", "r") as file:

    evaluation_set = json.load(file)

print("Evaluation dataset loaded successfully!")


# LOAD EMBEDDING MODEL
# ==========================================================
# Same embedding model used during ingestion.
#
# Important:
# Embedding model consistency is necessary because
# query embeddings and document embeddings must
# exist in the same vector space.

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# LOAD FAISS VECTOR STORE
# ==========================================================
# Loads previously generated vector index.

vector_store = FAISS.load_local(
    "vectorstores/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True,
)

print("FAISS index loaded successfully!")


# EVALUATION CONFIGURATION
# ==========================================================
# Top-K retrieval size
#
# Why K=5?
# ----------
# Common retrieval evaluation setting.
#
# Helps evaluate:
# - retrieval relevance
# - ranking quality
# - semantic search effectiveness

K = 5


# METRIC VARIABLES
# ==========================================================

total_queries = len(evaluation_set)

hit_count = 0

total_precision = 0
total_recall = 0
total_mrr = 0


# HELPER FUNCTION:
# CALCULATE MRR
# ==========================================================


def calculate_mrr(retrieved_crops, expected_crop):
    """
    Calculates Reciprocal Rank.

    MRR rewards systems that retrieve
    the correct answer earlier in ranking.

    Example:
    ----------
    Correct crop at rank 1 -> score = 1.0
    Correct crop at rank 2 -> score = 0.5
    Correct crop at rank 5 -> score = 0.2
    """

    for rank, crop in enumerate(retrieved_crops, start=1):

        if crop == expected_crop:
            return 1 / rank

    return 0


# EVALUATION LOOP
# ==========================================================
# Each query is tested against the retrieval pipeline.

for item in evaluation_set:

    question = item["question"]

    expected_crop = item["expected_crop"]

    # RETRIEVING TOP-K DOCUMENTS
    # ======================================================

    results = vector_store.similarity_search(question, k=K)

    retrieved_crops = [result.metadata["crop"] for result in results]

    # DISPLAY QUERY RESULTS
    # ======================================================

    print("\n===================================")
    print(f"Question: {question}")
    print(f"Expected Crop: {expected_crop}")
    print(f"Retrieved Crops: {retrieved_crops}")

    # HIT RATE@K
    # ======================================================
    # Measures whether correct crop exists
    # anywhere in top-K retrievals.

    if expected_crop in retrieved_crops:

        hit_count += 1

    # PRECISION@K
    # ======================================================
    # Measures:
    # How many retrieved results are relevant?

    relevant_retrieved = retrieved_crops.count(expected_crop)

    precision = relevant_retrieved / K

    total_precision += precision

    # RECALL@K
    # ======================================================
    # Since there is only one expected crop,
    # recall becomes:
    #
    # 1 -> crop retrieved
    # 0 -> crop not retrieved

    recall = 1 if expected_crop in retrieved_crops else 0

    total_recall += recall

    # MEAN RECIPROCAL RANK (MRR)
    # ======================================================

    reciprocal_rank = calculate_mrr(retrieved_crops, expected_crop)

    total_mrr += reciprocal_rank


# FINAL METRIC CALCULATIONS
# ==========================================================

hit_rate = hit_count / total_queries

average_precision = total_precision / total_queries

average_recall = total_recall / total_queries

mrr = total_mrr / total_queries


# DISPLAY FINAL RESULTS
# ==========================================================

print("\n===================================")
print("FINAL EVALUATION RESULTS")
print("===================================")

print(f"Hit Rate@{K}: {hit_rate:.2f}")

print(f"Precision@{K}: {average_precision:.2f}")

print(f"Recall@{K}: {average_recall:.2f}")

print(f"MRR: {mrr:.2f}")


# INTERPRETATION GUIDE
# ==========================================================
#
# Hit Rate:
# How often correct crop appeared in top-K.
#
# Precision:
# Retrieval purity / relevance quality.
#
# Recall:
# Ability to retrieve correct crop.
#
# MRR:
# Ranking quality.
# Higher MRR means correct crops appear earlier.
#
# ==========================================================
