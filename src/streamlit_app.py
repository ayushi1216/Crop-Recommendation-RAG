# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from rag_pipeline import process_query

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Crop Recommendation RAG System", page_icon="🌱", layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
<style>

/* Main background */
.stApp {
    background: linear-gradient(to bottom right, #f4fff4, #e8f5e9);
}

/* Main title */
.main-title {
    font-size: 3rem;
    font-weight: 700;
    color: #1b5e20;
    text-align: center;
    margin-bottom: 0.2rem;
}

/* Subtitle */
.subtitle {
    font-size: 1.1rem;
    color: #4e944f;
    text-align: center;
    margin-bottom: 2rem;
}

/* Input box */
.stTextInput > div > div > input {
    border-radius: 12px;
    border: 2px solid #81c784;
    padding: 12px;
    font-size: 16px;
}

/* Button */
.stButton > button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.7rem 2rem;
    font-size: 16px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #1b5e20;
    transform: scale(1.02);
}

/* Section headers */
.section-header {
    color: #1b5e20;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

/* Cards */
.custom-card {
    background-color: white;
    padding: 1.2rem;
    border-radius: 14px;
    border-left: 6px solid #43a047;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #edf7ed;
}

/* Footer */
.footer {
    text-align: center;
    color: #4e944f;
    margin-top: 3rem;
    font-size: 0.9rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🌱 Crop Recommendation RAG System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">AI-Powered Agricultural Recommendation Assistant using RAG</div>',
    unsafe_allow_html=True,
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌿 About")

st.sidebar.info("""
This system uses:

✅ Retrieval-Augmented Generation (RAG)

✅ Semantic Search using FAISS

✅ OpenAI Embeddings

✅ Crop-aware retrieval

✅ LLM-grounded answers

Built for intelligent agricultural recommendations.
""")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Example Questions")

example_questions = [
    "What crop grows best in high rainfall?",
    "Compare rice and maize nitrogen requirements",
    "Which crop is suitable for acidic soil?",
    "Best crop for humid tropical climate?",
]

for q in example_questions:
    st.sidebar.markdown(f"• {q}")

# ==========================================================
# USER INPUT
# ==========================================================

query = st.text_input(
    "🔍 Ask your agricultural question",
    placeholder="Example: Compare rice and maize nitrogen requirements",
)

# ==========================================================
# GENERATE BUTTON
# ==========================================================

if st.button("Generate Recommendation"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("🌾 Processing agricultural insights..."):

            result = process_query(query)

        # ==================================================
        # RETRIEVAL STRATEGY
        # ==================================================

        st.markdown(
            '<div class="section-header">🧠 Retrieval Strategy</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="custom-card">
            {result['strategy']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ==================================================
        # FINAL ANSWER
        # ==================================================

        st.markdown(
            '<div class="section-header">🤖 AI Recommendation</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="custom-card">
            {result['answer']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ==================================================
        # DETECTED CROPS
        # ==================================================

        if result["mentioned_crops"]:

            st.markdown(
                '<div class="section-header">🌱 Detected Crops</div>',
                unsafe_allow_html=True,
            )

            crop_text = ", ".join(result["mentioned_crops"])

            st.markdown(
                f"""
                <div class="custom-card">
                {crop_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ==================================================
        # RETRIEVED DOCUMENTS
        # ==================================================

        if result["retrieved_docs"]:

            st.markdown(
                '<div class="section-header">📄 Retrieved Context</div>',
                unsafe_allow_html=True,
            )

            for idx, doc in enumerate(result["retrieved_docs"], start=1):

                with st.expander(f"Retrieved Document {idx}"):

                    st.write(doc.page_content)

                    st.json(doc.metadata)

        # ==================================================
        # COMPARISON CONTEXT
        # ==================================================

        else:

            st.markdown(
                '<div class="section-header">📊 Aggregate Comparison Context</div>',
                unsafe_allow_html=True,
            )

            st.code(result["context"])

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
    🌾 Built with Streamlit, LangChain, FAISS, and OpenAI
    </div>
    """,
    unsafe_allow_html=True,
)
