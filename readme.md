# 🌱 Crop Recommendation RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system for intelligent crop recommendation and agricultural question answering using semantic search, vector databases, and Large Language Models (LLMs).


---

# 📌 Project Overview

Agricultural decision-making often depends on multiple environmental and soil-related factors such as:

- Nitrogen content
- Phosphorus levels
- Potassium levels
- Temperature
- Humidity
- Soil pH
- Rainfall

Traditional keyword-based search systems struggle to understand natural-language agricultural queries such as:

- *"What crop grows best with high rainfall and acidic soil?"*
- *"Compare rice and maize nitrogen requirements."*
- *"Which crop is suitable for humid tropical climate?"*

To solve this, this project implements a **Retrieval-Augmented Generation (RAG)** pipeline that:

1. Converts structured crop dataset rows into semantic documents
2. Generates embeddings using OpenAI Embedding Models
3. Stores embeddings in a FAISS vector database
4. Retrieves relevant agricultural context using semantic similarity
5. Generates grounded answers using an LLM

The system is designed to provide:
- context-aware recommendations
- grounded responses
- reduced hallucinations
- explainable retrieval-based outputs

---

# 🚀 Features

✅ Semantic Search using Vector Embeddings  
✅ Row-Level and Aggregate-Level Chunking  
✅ FAISS Vector Database  
✅ Crop-Aware Retrieval  
✅ Comparison Query Handling  
✅ Grounded LLM Responses  
✅ Retrieval Evaluation Metrics  
✅ Streamlit UI  
✅ Modular Backend Architecture  

---

# 🧠 System Architecture

## High-Level Pipeline

```text
CSV Dataset
     ↓
Data Ingestion
     ↓
Row/Aggregate Document Creation
     ↓
Embedding Generation
     ↓
FAISS Vector Store
     ↓
Semantic Retrieval
     ↓
Context Injection
     ↓
LLM Generation
     ↓
Final Grounded Answer
```

---

# ⚙️ Architecture Components

## 1. Data Ingestion

The original dataset is structured CSV data.

Since embedding models work best with natural language text, each row is converted into readable agricultural context.

Example:

```text
Crop: rice

Nitrogen: 90
Phosphorus: 42
Potassium: 43

Temperature: 20°C
Humidity: 82%
pH: 6.5
Rainfall: 202 mm
```

---

## 2. Embedding Generation

The system uses:

```python
text-embedding-3-small
```

### Why this model?

- strong semantic understanding
- cost-effective
- fast inference
- suitable for structured-to-text retrieval

Embeddings convert agricultural text into high-dimensional semantic vectors.

---

## 3. Vector Database

The project uses:

# FAISS (Facebook AI Similarity Search)

### Why FAISS?

- efficient vector similarity search
- lightweight local storage
- fast retrieval performance
- suitable for assignment-scale RAG systems

---

## 4. Retrieval Pipeline

The system supports multiple retrieval strategies:

### A. Generic Semantic Retrieval

Used when:
- no crop is explicitly mentioned
- query is condition-based

Example:

```text
What crop grows well in high rainfall?
```

The query embedding is matched semantically against vector embeddings.

---

### B. Crop-Aware Retrieval

Used when:
- a specific crop is mentioned

Example:

```text
Nitrogen requirement for rice
```

### Strategy:
1. Retrieve larger candidate pool
2. Apply metadata filtering
3. Keep only target crop documents

### Why?

Pure semantic retrieval may return unrelated crops with similar environmental conditions.

Crop-aware filtering improves retrieval precision.

---

### C. Aggregate Comparison Retrieval

Used for comparison queries such as:

```text
Compare rice and maize nitrogen requirements
```

Instead of retrieving individual rows, the system:
- aggregates crop statistics
- calculates average nutrient/environmental values
- generates summarized comparison context

This improves:
- clarity
- comparison quality
- retrieval relevance

---

# 📚 Chunking Strategies

The assignment required experimentation with multiple chunking approaches.

This project implements:

---

## 1. Row-Level Chunking

### Strategy

Each dataset row becomes one semantic document.

### Advantages

✅ Fine-grained retrieval  
✅ Better condition-level matching  
✅ More precise recommendations  

### Disadvantages

❌ Comparison queries become noisy  
❌ Larger vector store  

---

## 2. Aggregate-Level Chunking

### Strategy

All rows belonging to one crop are aggregated into summarized documents.

### Advantages

✅ Better for crop comparisons  
✅ Smaller vector database  
✅ Cleaner contextual summaries  

### Disadvantages

❌ Loses row-level granularity  
❌ Less effective for precise environmental matching  

---

# 🔍 Retrieval Experiments

The project experimented with:

| Strategy | Purpose |
|---|---|
| Row-Level Retrieval | Fine-grained recommendations |
| Aggregate Retrieval | Crop comparisons |
| Crop-Aware Filtering | Improve retrieval precision |
| Semantic Search | Natural language understanding |

---

# 🤖 Generation Layer

The system uses:

```python
gpt-4o-mini
```

for grounded answer generation.

The prompt is designed to:

✅ Use only retrieved context  
✅ Avoid hallucinations  
✅ Include supporting row IDs  
✅ Acknowledge insufficient context when needed  

---

# 📊 Evaluation

The system was evaluated using a manually created evaluation set consisting of crop-related question-answer pairs.

## Retrieval Metrics Implemented

| Metric | Purpose |
|---|---|
| Hit Rate@K | Measures if correct crop appears in top-K |
| Precision@K | Measures retrieval relevance quality |
| Recall@K | Measures ability to retrieve correct crop |
| MRR (Mean Reciprocal Rank) | Measures ranking quality |

---

# 📈 Evaluation Results

| Metric | Score |
|---|---|
| Hit Rate@5 | 0.72 |
| Precision@5 | 0.69 |
| Recall@5 | 0.52 |
| MRR | 0.59 |

---

# 🧪 Key Engineering Decisions

## Why Convert Structured Data into Text?

Embedding models work best with semantic natural language representations.

Converting rows into readable agricultural descriptions improves semantic retrieval quality.

---

## Why Use Metadata?

Metadata enables:
- crop-aware filtering
- row traceability
- grounded citations
- future hybrid retrieval expansion

---

## Why Use Multiple Retrieval Strategies?

Different query types require different retrieval approaches.

| Query Type | Best Strategy |
|---|---|
| Environmental condition queries | Semantic Retrieval |
| Crop-specific queries | Crop-Aware Retrieval |
| Comparison queries | Aggregate Retrieval |

---

# 🖥️ Streamlit UI

The project includes a modern Streamlit-based interface with:
- agricultural-themed UI
- semantic query input
- retrieval strategy display
- grounded answer generation
- retrieved context visualization

---

# 📁 Project Structure

```text
CROP_RAG_SYSTEM/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── Crop_recommendation.csv
│
├── evaluation/
│   └── evaluation_set.json
│
├── src/
│   ├── ingest.py
│   ├── ingest_aggregate.py
│   ├── rag_pipeline.py
│   ├── evaluate.py
│   └── compare_retrieval.py
│
├── vectorstores/
│   ├── faiss_row_index/
│   └── faiss_aggregate_index/
│
├── requirements.txt
├── README.md
└── .env
```

---

# 🛠️ Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd CROP_RAG_SYSTEM
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\\Scripts\\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add OpenAI API Key

Create `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

# ▶️ Running the Project

## Generate Row-Level Vector Store

```bash
python src/ingest.py
```

---

## Generate Aggregate Vector Store

```bash
python src/ingest_aggregate.py
```

---

## Run Evaluation

```bash
python src/evaluate.py
```

---

## Run Streamlit App

```bash
cd app
streamlit run streamlit_app.py
```

---

# 📌 Example Queries

```text
What crop grows best with high rainfall?
```

```text
Compare rice and maize nitrogen requirements
```

```text
Which crop is suitable for acidic soil?
```

```text
Best crop for humid tropical climate?
```


# 🏁 Conclusion

This project demonstrates the design and implementation of a production-style RAG pipeline for agricultural recommendation systems using:

- semantic retrieval
- vector databases
- LLM grounding
- retrieval evaluation
- modular AI system architecture

The system combines structured agricultural data with modern retrieval and generation techniques to provide context-aware, explainable crop recommendations.

---

# 👩‍💻 Author

Ayushi Prajapati