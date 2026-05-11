import pandas as pd
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# LOAD ENV VARIABLES
# ==========================================================

load_dotenv()


# LOAD DATASET
# ==========================================================
# Dataset required for:
# - comparison queries
# - aggregate statistics
# - crop-aware retrieval

df = pd.read_csv("../data/Crop_recommendation.csv")


# LOAD EMBEDDING MODEL
# ==========================================================

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# LOAD VECTOR STORE
# ==========================================================

vector_store = FAISS.load_local(
    "../vectorstores/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True,
)


# SUPPORTED CROPS
# ==========================================================

crop_names = [
    "rice",
    "maize",
    "chickpea",
    "kidneybeans",
    "pigeonpeas",
    "mothbeans",
    "mungbean",
    "blackgram",
    "lentil",
    "pomegranate",
    "banana",
    "mango",
    "grapes",
    "watermelon",
    "muskmelon",
    "apple",
    "orange",
    "papaya",
    "coconut",
    "cotton",
    "jute",
    "coffee",
]


# DETECT CROPS IN QUERY
# ==========================================================


def detect_crops(query):
    """
    Detect crop names mentioned in query.

    Why?
    -----
    Helps implement crop-aware retrieval.
    """

    query_lower = query.lower()

    detected = []

    for crop in crop_names:

        if crop in query_lower:
            detected.append(crop)

    return detected


# BUILD COMPARISON CONTEXT
# ==========================================================


def build_comparison_context(crops):
    """
    Generate aggregate comparison context.

    Used for:
    - crop comparison queries
    - summarized trend analysis
    """

    comparison_context = ""

    for crop in crops:

        crop_rows = df[df["label"] == crop]

        avg_n = crop_rows["N"].mean()
        avg_p = crop_rows["P"].mean()
        avg_k = crop_rows["K"].mean()

        avg_temp = crop_rows["temperature"].mean()
        avg_ph = crop_rows["ph"].mean()
        avg_rainfall = crop_rows["rainfall"].mean()

        comparison_context += f"""
Crop: {crop}

Average Nitrogen: {avg_n:.2f}
Average Phosphorus: {avg_p:.2f}
Average Potassium: {avg_k:.2f}

Average Temperature: {avg_temp:.2f} °C
Average pH: {avg_ph:.2f}
Average Rainfall: {avg_rainfall:.2f} mm
"""

    return comparison_context


# SINGLE CROP RETRIEVAL
# ==========================================================


def retrieve_single_crop_context(query, target_crop):
    """
    Semantic retrieval + metadata filtering.

    Why?
    -----
    Improves retrieval precision by filtering
    only relevant crop rows.
    """

    all_results = vector_store.similarity_search(query, k=20)

    filtered_results = []

    for result in all_results:

        crop = result.metadata["crop"]

        if crop == target_crop:
            filtered_results.append(result)

        if len(filtered_results) >= 3:
            break

    context = ""

    for result in filtered_results:

        context += f"""
[Row ID: {result.metadata['row_id']}]
{result.page_content}
"""

    return context, filtered_results


# GENERIC SEMANTIC RETRIEVAL
# ==========================================================


def retrieve_generic_context(query):
    """
    Generic dense-vector semantic retrieval.
    """

    results = vector_store.similarity_search(query, k=5)

    context = ""

    for result in results:

        context += f"""
[Row ID: {result.metadata['row_id']}]
{result.page_content}
"""

    return context, results


# GENERATING FINAL ANSWER
# ==========================================================


def generate_answer(query, context):
    """
    Generate grounded LLM response.
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
You are an agricultural recommendation assistant.

Answer ONLY using the retrieved context below.

Do not make assumptions outside the dataset.

If the context is insufficient or out of dataset, say:
"I do not have enough information from the dataset."

Retrieved Context:
{context}

Question:
{query}

Provide:
1. Recommended crop or comparison
2. Clear explanation
"""

    response = llm.invoke(prompt)
    return response.content


# MAIN PROCESSING FUNCTION
# ==========================================================


def process_query(query):
    """
    Main reusable backend pipeline.

    This function can be used by:
    - Streamlit UI
    - CLI applications
    - APIs
    - future integrations
    """

    mentioned_crops = detect_crops(query)

    retrieved_docs = []

    # COMPARISON QUERY
    # ======================================================

    if len(mentioned_crops) >= 2:

        strategy_used = "Aggregate Comparison Retrieval"

        context = build_comparison_context(mentioned_crops)

    # SINGLE CROP QUERY
    # ======================================================

    elif len(mentioned_crops) == 1:

        strategy_used = "Semantic Retrieval + Metadata Filtering"

        target_crop = mentioned_crops[0]

        context, retrieved_docs = retrieve_single_crop_context(query, target_crop)

    # GENERIC QUERY
    # ======================================================

    else:

        strategy_used = "Genaral Semantic Retrieval"

        context, retrieved_docs = retrieve_generic_context(query)

    # GENERATE FINAL ANSWER
    # ======================================================

    final_answer = generate_answer(query, context)

    # RETURN RESULTS
    # ======================================================

    return {
        "query": query,
        "strategy": strategy_used,
        "mentioned_crops": mentioned_crops,
        "context": context,
        "retrieved_docs": retrieved_docs,
        "answer": final_answer,
    }


# OPTIONAL CLI MODE
# ==========================================================


def run_cli():
    """
    Optional CLI execution.
    """

    query = input("\nAsk your crop question: ")

    result = process_query(query)

    print("\nStrategy Used:")
    print(result["strategy"])

    print("\nFinal Answer:\n")
    print(result["answer"])


# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    run_cli()
