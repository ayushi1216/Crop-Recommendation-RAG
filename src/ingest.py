import pandas as pd
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# LOAD ENV VARIABLES
# ==========================================================
# Loads OpenAI API key from .env

load_dotenv()

# LOAD DATASET
# ==========================================================
# Dataset contains:
# - Soil nutrients
# - Environmental conditions
# - Recommended crop label

df = pd.read_csv("data/Crop_recommendation.csv")

print("Crop dataset loaded successfully!")

# HELPER FUNCTION:
# CONVERT CSV ROWS TO DOCUMENTS
# ==========================================================


def create_documents(dataframe):
    """
    Converts structured CSV rows into natural-language documents.

    Strategy Used:
    ----------------
    Row-Level Chunking

    Why?
    -----
    Each row represents one unique crop-growing condition.

    Converting rows into text helps embedding models
    understand the semantic relationship between:
    - soil nutrients
    - weather conditions
    - crop suitability

    Example Converted Document:
    ----------------------------
    Crop: rice
    Nitrogen: 90
    Rainfall: 200 mm
    pH: 6.5

    Benefits:
    ----------
    - Fine-grained retrieval
    - Better semantic matching
    - Supports condition-based queries
    """

    documents = []

    for idx, row in dataframe.iterrows():

        # Convert structured row into natural language text
        text = f"""
Crop: {row['label']}

Soil Nutrients:
Nitrogen: {row['N']}
Phosphorus: {row['P']}
Potassium: {row['K']}

Environmental Conditions:
Temperature: {row['temperature']} °C
Humidity: {row['humidity']} %
pH: {row['ph']}
Rainfall: {row['rainfall']} mm
"""

        # Metadata helps in:
        # - filtering
        # - traceability
        # - citation grounding
        # - crop-aware retrieval

        doc = Document(
            page_content=text,
            metadata={
                "crop": row["label"],
                "row_id": int(idx),
                "ph": float(row["ph"]),
                "rainfall": float(row["rainfall"]),
            },
        )

        documents.append(doc)

    return documents


# CREATE DOCUMENTS
# ==========================================================

documents = create_documents(df)

print(f"Total documents created: {len(documents)}")


# LOAD EMBEDDING MODEL
# ==========================================================
# Why text-embedding-3-small?
#
# - Lightweight and fast
# - Strong semantic understanding
# - Cost-effective
# - Suitable for structured RAG systems

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# CREATE FAISS VECTOR STORE
# ==========================================================
# Why FAISS?
#
# - Efficient similarity search
# - Fast vector retrieval
# - Runs locally
# - Good for experimentation and assignments

vector_store = FAISS.from_documents(documents, embedding_model)

print("FAISS vector store created successfully!")


# SAVE VECTOR STORE
# ==========================================================
# Persist index locally for reuse

vector_store.save_local("vectorstores/faiss_index")

print("FAISS index saved locally!")


# VALIDATE RETRIEVAL PIPELINE
# ==========================================================
# Small retrieval test to verify:
# - embeddings are working
# - vector store is functional
# - semantic retrieval is meaningful

loaded_vector_store = FAISS.load_local(
    "vectorstores/faiss_index", embedding_model, allow_dangerous_deserialization=True
)

query = "crop suitable for high rainfall"

results = loaded_vector_store.similarity_search(query, k=2)

print("\nSample Retrieval Results:\n")

for idx, result in enumerate(results, 1):

    print(f"Result {idx}")
    print(result.page_content)
    print()


# OPTIONAL:
# CHROMA DB IMPLEMENTATION (EXPERIMENTAL)
# ==========================================================
#
# ChromaDB was also explored as an alternative vector DB.
#
# Why experiment with multiple vector stores?
#
# - Compare retrieval quality
# - Compare persistence mechanisms
# - Compare developer experience
#
# FAISS was finally chosen because:
# - simpler local setup
# - faster experimentation
# - lightweight for assignment-scale system

# from langchain_chroma import Chroma
#
# vector_store = Chroma(
#     collection_name="crop_collection",
#     embedding_function=embedding_model,
#     persist_directory="chromadb_store",
# )
#
# Batch insertion can help with:
# - memory optimization
# - large dataset handling
#
# batch_size = 100
#
# for i in range(0, len(documents), batch_size):
#
#     batch = documents[i : i + batch_size]
#
#     vector_store.add_documents(batch)
#
# print("All embeddings stored successfully!")
