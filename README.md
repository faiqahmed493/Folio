# 📚 Folio — RAG Based Document Intelligence System

> Transform static documents into interactive conversational knowledge bases using Retrieval-Augmented Generation (RAG).

---

## ✨ Overview

**Folio** is a modern **Retrieval-Augmented Generation (RAG)** application built using **Streamlit**, **LangChain**, and **ChromaDB**. It enables users to upload documents and interact with them conversationally using natural language queries.

Unlike generic AI chatbots, Folio provides **source-grounded answers only from the uploaded document**. If the answer is not present in the source, the system explicitly states that the information could not be found.

This makes Folio ideal for:

- 📖 Academic Research
- ⚖️ Legal Document Review
- 📘 Technical Manual Analysis
- 📝 Literary Study
- 📊 Knowledge Extraction

---

# 🚀 Features

## 📄 Document Indexing

- Upload and process documents in seconds
- Semantic chunking with overlap preservation
- Persistent vector database storage
- Supports:
  - PDF
  - DOCX
  - PPT
  - TXT
  - CSV

---

## 💬 Multi-turn Conversational QA

- Context-aware follow-up questioning
- Maintains conversation history
- Enables iterative exploration of documents

---

## 🎯 Confidence Scoring

- Retrieval confidence estimation (0–100%)
- Color-coded confidence indicators
- Helps users assess answer reliability

---

## 📚 Readability Analysis

Computes **Flesch Reading Ease Score** and classifies documents into:

- 🟢 Easy
- 🟡 Medium
- 🟠 Hard
- 🔴 Very Hard

---

## 📝 Summary Generation

- Generates concise executive summaries
- Uses focused retrieval for introduction-related sections

---

## 📌 Answer Pinning & Export

- Pin important answers
- Export conversations and saved responses

---

# 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Frontend | Streamlit |
| Framework | LangChain |
| Vector Database | ChromaDB |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| LLM Providers | Groq, Gemini, Mistral |
| Language | Python 3.10+ |

---

# 🧠 System Architecture

```text
Document Upload
       ↓
Document Parsing
       ↓
Text Chunking
       ↓
Embedding Generation
       ↓
ChromaDB Vector Storage
       ↓
MMR Retrieval
       ↓
LLM Response Generation
       ↓
Source-Grounded Answer
```

---

# ⚙️ Methodology

## 1️⃣ Document Ingestion

Documents are parsed page-by-page using LangChain document loaders.

### Chunking Strategy

- **Chunk Size:** 1000 characters
- **Overlap:** 200 characters

The overlap preserves contextual continuity between chunks.

---

## 2️⃣ Embedding & Indexing

Each chunk is converted into dense vector embeddings using:

```python
sentence-transformers/all-MiniLM-L6-v2
```

### Features

- 384-dimensional embeddings
- Semantic similarity retrieval
- Persistent local vector storage

---

## 3️⃣ Retrieval

Folio uses **Maximal Marginal Relevance (MMR)** retrieval.

### Retrieval Configuration

| Parameter | Value |
|---|---|
| k | 5 |
| fetch_k | 12 |
| lambda | 0.5 |

MMR balances:

- Relevance
- Diversity

This prevents redundant retrieval results.

---

## 4️⃣ Answer Generation

Retrieved passages are combined with:

- Conversation history
- Prompt templates
- Selected LLM backend

### Supported Models

- 🦙 Groq (LLaMA 3.1-8B)
- ✨ Gemini 2.5 Flash
- 🌪️ Mistral 7B

The system prompt strictly enforces:

- Source-grounded responses
- Hallucination prevention
- Explicit fallback when information is unavailable

---

# 🎨 User Interface

## 📌 Sidebar

- LLM selector
- Pinned answers
- Conversation history
- Export controls

---

## 📄 Main Panel

- Document upload
- Question input
- Confidence visualization
- Retrieved source references
- Summary generation

---

# 📊 Analytics & Visualization

## 📈 Confidence Bar

Animated progress bar displaying retrieval confidence.

---

## 📊 Stat Cards

Displays:

- Page count
- Chunk count
- Average chunk length
- Readability score

---

## 🏷️ Source Chips

Displays page-level provenance of retrieved passages.

---

## 🚦 Difficulty Badge

Visual readability indicator based on document complexity.

---

# 📂 Project Structure

```text
folio/
│
├── app.py
├── embeddings/
├── prompts/
├── vectorstore/
├── utils/
├── styles/
├── exports/
└── requirements.txt
```

---

# 🧩 Core Modules

| Module | Responsibility |
|---|---|
| Theme Engine | UI styling |
| CSS Engine | Streamlit customization |
| Embeddings | Embedding generation |
| LLM Factory | Backend model selection |
| Analytics | Confidence & readability |
| Prompt Templates | QA & summarization |
| Retry Logic | API retry/backoff handling |

---

# ⚡ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/folio.git
cd folio
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_key
GOOGLE_API_KEY=your_key
MISTRAL_API_KEY=your_key
```

---

## 4️⃣ Run Application

```bash
streamlit run app.py
```

---

# 🧪 Example Workflow

1. Upload a document
2. Build vector index
3. Ask natural language questions
4. Receive grounded answers
5. Pin/export important insights

---

# 📌 Performance Insights

- MMR retrieval improves retrieval diversity
- 1000-character chunking balances context and precision
- Groq backend provides the lowest latency (~1–3 seconds)
- Persistent ChromaDB storage avoids unnecessary re-indexing

---

# 🔮 Future Work

- 📚 Multi-document RAG
- 🎯 Cross-encoder re-ranking
- 👤 User authentication
- 🔖 Citation footnotes
- 🌐 REST API support
- 📊 RAGAS evaluation dashboard

---

# 📖 References

1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS 2020)
2. LangChain Documentation
3. ChromaDB Documentation
4. Sentence-BERT (EMNLP 2019)
5. Streamlit Documentation
6. Groq API Documentation
7. Gemini API Documentation
8. Mistral AI Documentation

---

# 👨‍💻 Author

Developed as a university project for the Department of Computer Science & Information Technology.
