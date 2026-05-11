import pandas as pd
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# LOAD ENV VARIABLES
# ==========================================================
# Loads OpenAI API key from .env file

load_dotenv()


# LOAD DATASET
# ==========================================================
# Dataset contains crop recommendation data
# with:
# - soil nutrients
# - environmental conditions
# - crop labels

df = pd.read_csv("data/Crop_recommendation.csv")

print("Crop dataset loaded successfully!")


# HELPER FUNCTION:
# CREATE AGGREGATE DOCUMENTS
# ==========================================================


def create_aggregate_documents(dataframe):
    """
    Creates one summarized document per crop.

    Strategy Used:
    ----------------
    Aggregate Chunking

    Why?
    -----
    Instead of storing every row individually,
    we summarize all rows belonging to a crop.

    This helps in:
    - comparison queries
    - overall crop analysis
    - reducing retrieval noise
    - compact context generation

    Example:
    --------
    "Compare rice and maize nitrogen requirements"

    Aggregate retrieval performs better because:
    - users want overall trends
    - average values are more meaningful
    - avoids redundant row-level retrieval

    Tradeoff:
    ----------
    Pros:
    - cleaner comparisons
    - smaller vector store
    - faster retrieval

    Cons:
    - loses fine-grained row-level detail
    - less precise for condition-specific queries
    """

    documents = []

    # Group dataset by crop label
    grouped = dataframe.groupby("label")

    for crop, group in grouped:

        # CALCULATE AGGREGATE STATISTICS
        # ==================================================

        avg_n = group["N"].mean()
        avg_p = group["P"].mean()
        avg_k = group["K"].mean()

        avg_temp = group["temperature"].mean()
        avg_humidity = group["humidity"].mean()

        avg_ph = group["ph"].mean()
        avg_rainfall = group["rainfall"].mean()

        # Range values help capture variability
        min_temp = group["temperature"].min()
        max_temp = group["temperature"].max()

        min_ph = group["ph"].min()
        max_ph = group["ph"].max()

        # CONVERT AGGREGATED DATA INTO NATURAL LANGUAGE
        # ==================================================

        text = f"""
Crop: {crop}

Average Soil Nutrients:
Nitrogen: {avg_n:.2f}
Phosphorus: {avg_p:.2f}
Potassium: {avg_k:.2f}

Average Environmental Conditions:
Temperature: {avg_temp:.2f} °C
Humidity: {avg_humidity:.2f} %
pH: {avg_ph:.2f}
Rainfall: {avg_rainfall:.2f} mm

Typical Temperature Range:
{min_temp:.2f} °C to {max_temp:.2f} °C

Typical pH Range:
{min_ph:.2f} to {max_ph:.2f}
"""

        # Metadata can help in:
        # - filtering
        # - retrieval tracking
        # - future hybrid retrieval

        doc = Document(page_content=text, metadata={"crop": crop, "type": "aggregate"})

        documents.append(doc)

    return documents


# CREATE AGGREGATE DOCUMENTS
# ==========================================================

documents = create_aggregate_documents(df)

print(f"Total aggregate documents created: {len(documents)}")


# LOAD EMBEDDING MODEL
# ==========================================================
# Same embedding model used for consistency
# across experiments.

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

print("Generating aggregate embeddings...")


# CREATE FAISS VECTOR STORE
# ==========================================================
# Aggregate documents are converted into embeddings
# and indexed for semantic retrieval.

vector_store = FAISS.from_documents(documents, embedding_model)

print("Aggregate FAISS vector store created successfully!")


# SAVE VECTOR STORE
# ==========================================================
# Store locally for reuse in retrieval pipeline

vector_store.save_local("vectorstores/faiss_aggregate_index")

print("Aggregate FAISS index saved successfully!")


# VALIDATE RETRIEVAL PIPELINE
# ==========================================================
# Small test query to verify:
# - embedding generation
# - semantic retrieval
# - aggregate chunk effectiveness

query = "Compare rice and maize nitrogen requirements"

results = vector_store.similarity_search(query, k=2)

print("\nSample Retrieval Results:\n")

for idx, result in enumerate(results, 1):

    print(f"Result {idx}")
    print(result.page_content)
    print()
