<p align="center">
  <img src="docs/banner.svg" width="100%" />
</p>

# 🌍 Travel RAG Assistant

📚 Retrieval-Augmented Generation System for Tourism Information

LangChain • Gemini 2.5 Flash • ChromaDB • Pinecone • Streamlit

## 🌐 Live Demo

> Deployment in progress. Check back soon for the live application!

<p align="center">
  <!-- Environment / Tooling -->
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Poetry-1.8+-6A5ACD?logo=poetry" />
  <img src="https://img.shields.io/badge/LangChain-Latest-00A67E" />
  <img src="https://img.shields.io/badge/Gemini-Pro-4285F4?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Supported-FF6F00" />
  <img src="https://img.shields.io/badge/Pinecone-Cloud-00C9A7" />
  <img src="https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

# 📑 Table of Contents

- [Overview](#overview)
- [Current Dataset](#-current-dataset)
- [Tech Stack](#-tech-stack)
- [Project Layout](#-project-layout)
- [Quick Start](#-quick-start)
- [Notebook / Chapter Overview](#-notebook--chapter-overview)
- [RAG Pipeline Architecture](#-rag-pipeline-architecture)
- [Future Work](#-future-work)
- [License](#-license)

---

## Overview

This project builds a complete Retrieval-Augmented Generation (RAG) system for tourism information using modern LLM technologies.

**Current Status:** Conversational RAG with Citations (Chapters 01-05 completed)

**Goal:** Create an AI-powered travel assistant that can answer questions about Seattle tourism by retrieving relevant information from a vector database and generating natural language responses using Google Gemini.

**Key Components:**

- Geoapify API for attraction data
- Wikipedia API for detailed descriptions
- Vector database for semantic search
- LangChain for RAG orchestration  
- Streamlit for interactive web interface

## 📊 Current Dataset

- **Cities**: 13 major US cities
  - Seattle, New York, Washington DC, Chicago, San Francisco
  - Boston, Portland, Austin, Denver, Miami
  - Nashville, New Orleans, Las Vegas
- **Primary Source**: [Geoapify Places API](https://www.geoapify.com/)
- **Secondary Source**: [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)
- **Total Records**: ~1,000+ attractions with Wikipedia descriptions
- **Largest Dataset**: Washington DC (264 attractions)
- **Format**: JSON (enriched with location data and descriptions)
- **Fields**: Name, Description, Location (lat/lon), Address, Categories, Place ID, City, State, Country

## 🛠️ Tech Stack

### **Core Components**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Gemini 2.5 Flash | Text generation and question answering |
| **Embedding** | sentence-transformers (all-MiniLM-L6-v2) | Document vectorization (384 dimensions) |
| **Vector DB (Dev)** | ChromaDB | Local vector storage and retrieval |
| **Vector DB (Prod)** | Pinecone | Cloud-based vector database |
| **RAG Framework** | LangChain | Pipeline orchestration |
| **Frontend** | Streamlit | Interactive web interface |
| **Data Processing** | Pandas | Data manipulation and cleaning |

### **Development Tools**

- **Python**: 3.13
- **Package Manager**: Poetry
- **Environment**: python-dotenv
- **Logging**: Custom emoji logger

---

## 📁 Project Layout

```text
.
├─ data/
│  ├─ raw/                      # Raw API responses
│  │  └─ Seattle_attractions_raw.json
│  └─ processed/                # Processed & enriched data
│     ├─ seattle_attractions_with_wikipedia.json
│     ├─ seattle_attractions_enriched_with_location.json
│     ├─ seattle_attractions_documents.json
│     └─ metadata.json
│
├─ chroma_db/                   # ChromaDB vector storage (local)
│
├─ notebook/                    # Jupyter Notebooks (exploration)
│  ├─ 01_data_exploration.ipynb      # Geoapify API & data collection
│  ├─ 02_data_enrichment.ipynb       # Wikipedia descriptions & location data
│  ├─ 03_vector_database.ipynb       # ChromaDB setup & semantic search
│  ├─ 04_rag_pipeline.ipynb          # RAG pipeline with Gemini LLM
│  └─ 05_conversational_rag.ipynb    # Conversation memory & source citations
│
├─ src/
│  ├─ app/                      # Streamlit web application
│  │  └─ app.py
│  ├─ data_collection/          # Data pipeline modules
│  │  ├─ geoapify_client.py    # Geoapify API client
│  │  ├─ wikipedia_client.py   # Wikipedia API client
│  │  ├─ collector.py          # Chapter 1 workflow
│  │  ├─ enricher.py           # Chapter 2 workflow
│  │  └─ document_builder.py   # RAG document formatting
│  ├─ rag/                      # RAG pipeline implementation
│  │  ├─ embeddings.py         # Embedding model management
│  │  ├─ vector_store.py       # ChromaDB vector store operations
│  │  ├─ llm.py                # LLM (Gemini) management
│  │  ├─ prompts.py            # Prompt template management
│  │  └─ rag_chain.py          # RAG chain assembly
│  ├─ utils/                    # Utilities
│  │  ├─ logger.py
│  │  └─ emoji_log.py          # Emoji-enhanced logging
│  └─ config.py                 # Configuration management
│
├─ scripts/                     # Utility scripts
│  ├─ ingest_data.py           # Data collection automation
│  ├─ setup_chromadb.py        # Vector database setup
│  └─ test_rag.py              # RAG system testing
│
├─ .env.example                 # Environment variables template
├─ .gitignore
├─ pyproject.toml
├─ poetry.lock
└─ README.md
```

---

## 🚀 Quick Start

### **1. Clone Repository**

```bash
git clone https://github.com/chgwyellow/travel_rag.git
cd travel_rag
```

### **2. Install Dependencies**

```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### **3. Set Up Environment Variables**

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
GEOAPIFY_API_KEY=your_geoapify_api_key_here
EMAIL=your_email@example.com  # For Wikipedia API User-Agent
GOOGLE_API_KEY=your_gemini_api_key_here  # For future RAG pipeline
```

### **4. Run Data Collection Pipeline**

Execute the automated data pipeline to collect and enrich Seattle attractions:

```bash
# Run the complete pipeline (Chapters 1-2)
poetry run python scripts/ingest_data.py
```

This will:

- Fetch attractions from Geoapify API
- Filter attractions with Wikipedia links
- Enrich with Wikipedia descriptions
- Create RAG-ready documents

**Output files** (in `data/`):

- `raw/Seattle_attractions_raw.json` - Raw Geoapify data
- `processed/Seattle_attractions_with_wikipedia.json` - Filtered attractions
- `processed/Seattle_attractions_enriched.json` - Enriched with descriptions
- `processed/Seattle_attractions_documents.json` - RAG-ready documents

### **5. Build Vector Database**

After collecting and enriching the data, build the ChromaDB vector database:

```bash
# Build vector database for Seattle (default)
poetry run python scripts/setup_chromadb.py

# Or specify a different city
poetry run python scripts/setup_chromadb.py --city "New York"

# Custom collection name
poetry run python scripts/setup_chromadb.py --city Seattle --collection my_attractions
```

This will:

- Load processed documents from `data/processed/`
- Create HuggingFace embedding model (all-MiniLM-L6-v2)
- Initialize ChromaDB persistent client
- Generate 384-dimensional embeddings for all documents
- Store documents with metadata in ChromaDB

**Output:**

- `chroma_db/` directory with vector database
- Ready for semantic search and RAG pipeline

### **6. Test RAG System**

Test the complete RAG question-answering pipeline:

```bash
# Test with default question
poetry run python scripts/test_rag.py

# Test with custom question
poetry run python scripts/test_rag.py --question "What are some museums in Seattle?"

# Test location-based query
poetry run python scripts/test_rag.py --question "Tell me about attractions near Pike Place Market"
```

This will:

- Load vector store and embedding model
- Initialize Google Gemini LLM
- Build RAG chain with prompt template
- Execute semantic search and generate answer
- Display natural language response

### **7. Explore with Jupyter Notebooks**

```bash
# Start Jupyter to explore the data collection process
poetry run jupyter lab

# Open notebooks:
# - 01_data_exploration.ipynb (Geoapify API exploration)
# - 02_data_enrichment.ipynb (Wikipedia enrichment process)
# - 03_vector_database.ipynb (ChromaDB setup)
# - 04_rag_pipeline.ipynb (RAG pipeline with LLM)
```

### **8. Run Streamlit App (Coming Soon)**

After completing RAG pipeline implementation:

```bash
poetry run streamlit run src/app/app.py
```

---

## 📓 Notebook / Chapter Overview

<details>
<summary><b>📊 Chapter 01 — Data Exploration (Geoapify API)</b></summary>

📓 `01_data_exploration.ipynb`

**Objectives:**

- Set up Geoapify API for Seattle attractions
- Fetch tourism data within Seattle bounding box
- Filter attractions with Wikipedia links
- Analyze data structure and quality
- Design document format for RAG

**Implementation:**

- Geoapify Places API with `tourism` category filter
- Bounding box: Seattle metropolitan area
- Filtered for attractions with Wikipedia data
- Saved raw data to `data/raw/Seattle_attractions_raw.json`

**Key Findings:**

- 62 attractions with Wikipedia links (from ~500 total)
- All attractions have place_id, name, and location data
- Wikipedia codes in format "language:title" (e.g., "en:Space Needle")
- Categories include landmarks, museums, parks, monuments

**Output:**

- `seattle_attractions_with_wikipedia.json` - 62 filtered attractions
- Document format design for RAG
- Ready for Wikipedia enrichment

</details>

---

<details>
<summary><b>✨ Chapter 02 — Data Enrichment (Wikipedia API)</b></summary>

📓 `02_data_enrichment.ipynb`

**Objectives:**

- Fetch Wikipedia descriptions for all 62 attractions
- Merge location data from raw Geoapify response
- Perform data quality analysis and cleaning
- Create RAG-ready document format
- Validate data completeness

**Implementation:**

- Wikipedia API with User-Agent header
- Batch fetching with 0.5s rate limiting
- Location data merge (lat, lon, address, city, state, postcode)
- Document format: Name + Location + Coordinates + Description

**Data Quality Results:**

- ✅ 62/62 attractions with descriptions (100% success)
- ✅ 0 duplicates (based on place_id)
- ✅ 27.4% descriptions contain special characters (normal)
- ✅ Average description length: 860 characters
- ✅ 100% data completeness

**Output:**

- `seattle_attractions_enriched_with_location.json` - Full enriched data
- `seattle_attractions_documents.json` - RAG-ready documents
- `metadata.json` - Updated with enrichment statistics
- Ready for vector database ingestion

</details>

---

<details>
<summary><b>🔍 Chapter 03 — Vector Database Setup (ChromaDB)</b></summary>

📓 `03_vector_database.ipynb`

**Objectives:**

- Set up ChromaDB and embedding models
- Generate embeddings for all attraction documents
- Implement semantic search functionality
- Test metadata filtering capabilities
- Validate retrieval quality

**Implementation:**

- HuggingFace embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- ChromaDB PersistentClient for vector storage
- 384-dimensional embeddings for all documents
- Semantic similarity search with score analysis

**Quality Results:**

- ✅ 62/62 documents indexed successfully
- ✅ 100% test pass rate in quality validation
- ✅ Semantic search working correctly
- ✅ Average query time: <100ms
- ✅ Database size: ~50 MB

**Output:**

- `chroma_db/` - Persistent vector database storage
- `chroma_db/travel_attractions/` - Collection with embeddings
- Ready for RAG pipeline integration with LLM

</details>

---

<details>
<summary><b>🤖 Chapter 04 — RAG Pipeline with LLM</b></summary>

📓 `04_rag_pipeline.ipynb`

**Objectives:**

- Integrate Google Gemini 2.5 Flash LLM
- Build complete RAG question-answering pipeline
- Design effective prompt templates
- Test and evaluate response quality

**Implementation:**

- **LLM**: Google Gemini 2.0 Flash (experimental)
- **Retriever**: ChromaDB similarity search (k=5)
- **Prompt Engineering**: Travel assistant with context-based answering
- **Chain**: LangChain Expression Language (LCEL) pipeline

**Modules Created:**

- `src/rag/llm.py` - LLM management functions
- `src/rag/prompts.py` - Prompt template management
- `src/rag/rag_chain.py` - RAG chain assembly
- `scripts/test_rag.py` - RAG testing script

**Test Results:**

- ✅ Basic Q&A: Accurate answers for simple questions
- ✅ Complex queries: Successfully handled multi-part questions
- ✅ Out-of-scope: Correctly refused questions outside database
- ✅ Location queries: Found nearby attractions effectively

**Example Usage:**

```bash
# Test RAG system with default question
poetry run python scripts/test_rag.py

# Test with custom question
poetry run python scripts/test_rag.py --question "What are museums in Seattle?"
```

**Output:**

- Functional RAG pipeline ready for deployment
- Natural language responses based on vector database
- Source-grounded answers (no hallucination)

</details>

---

<details>
<summary><b>💬 Chapter 05 — Conversation Memory & Source Citations</b></summary>

📓 `05_conversational_rag.ipynb`

**Objectives:**

- Add conversation memory for multi-turn dialogues
- Implement source citations for transparency
- Enhance RAG system with context awareness
- Format citations for easy verification

**Implementation:**

- **Memory**: `ChatMessageHistory` with session management
- **Context**: `MessagesPlaceholder` for chat history in prompts
- **Data Flow**: `RunnablePassthrough.assign()` to preserve chat history
- **Citations**: `RunnableParallel` to return both answers and sources
- **Formatting**: Detailed citation formatter with metadata

**Key Components:**

- `RunnablePassthrough.assign()` - Preserves chat history while adding retrieved docs
- `RunnableParallel` - Returns answer + source documents simultaneously
- `RunnableWithMessageHistory` - Manages conversation sessions with `output_messages_key`
- `format_citations_detailed()` - Formats sources for display
- Session-based history storage

**Features:**

- ✅ Multi-turn conversations with context memory
- ✅ Understands pronouns and references (e.g., "it", "there")
- ✅ Provides verifiable source documents
- ✅ Formatted citations with location and metadata
- ✅ Session isolation for multiple users

**Example:**

```python
# Turn 1
result = conversational_chain_with_history_and_sources.invoke(
    {"question": "What is the Space Needle?"},
    config={"configurable": {"session_id": "user_123"}}
)

# Turn 2 (remembers context)
result = conversational_chain_with_history_and_sources.invoke(
    {"question": "How tall is it?"},  # "it" = Space Needle
    config={"configurable": {"session_id": "user_123"}}
)

# Display answer and sources
print(result["answer"])
print(format_citations_detailed(result["source_documents"]))
```

**Technical Note:**

The implementation uses `RunnablePassthrough.assign()` instead of custom functions to ensure `chat_history` is preserved throughout the chain. This is critical for conversation memory to work correctly. The `output_messages_key="answer"` parameter tells `RunnableWithMessageHistory` where to find the answer in the output dictionary for storage.

**Output:**

- Conversational RAG system with memory
- Source citations for every answer
- Ready for Streamlit deployment

</details>

---

## 🧩 RAG Pipeline Architecture

```mermaid
flowchart TD

%% ============================
%% COLOR THEMES
%% ============================
classDef data fill:#BBDEFB,stroke:#0D47A1,stroke-width:2px,color:#0D47A1,rx:10,ry:10
classDef process fill:#DCEDC8,stroke:#33691E,stroke-width:2px,color:#1B5E20,rx:10,ry:10
classDef rag fill:#FFE0B2,stroke:#E65100,stroke-width:2px,color:#E65100,rx:10,ry:10
classDef llm fill:#F8BBD0,stroke:#AD1457,stroke-width:2px,color:#880E4F,rx:10,ry:10
classDef ui fill:#D1C4E9,stroke:#4527A0,stroke-width:2px,color:#311B92,rx:10,ry:10

%% ============================
%% NODES
%% ============================

A["📊 Raw Tourism Data<br>JSON from Open Data"]

subgraph DATA[Data Processing]
    B1["🧹 Data Cleaning"]
    B2["📝 Document Creation<br>Merge Fields"]
    B3["✂️ Text Chunking<br>500 tokens, overlap 50"]
end

subgraph EMBED[Embedding & Storage]
    C1["🤖 SentenceTransformer<br>all-MiniLM-L6-v2<br>384-dim Vectors"]
    C2["💾 Vector Database<br>ChromaDB / Pinecone"]
end

D["👤 User Query<br>Natural Language Question"]

subgraph RAG[RAG Pipeline]
    E1["🔍 Query Embedding<br>SentenceTransformer"]
    E2["🎯 Similarity Search<br>Top-K Retrieval"]
    E3["📄 Context Assembly<br>Retrieved Documents"]
end

subgraph LLM[LLM Generation]
    F1["💬 Prompt Construction<br>Context + Query"]
    F2["🧠 Gemini 2.5 Flash<br>Response Generation"]
end

G["✨ AI Response<br>+ Source Citations"]

subgraph UI[User Interface]
    H1["🎨 Streamlit App<br>Interactive UI"]
    H2["💬 Conversation History"]
end

%% ============================
%% FLOWS
%% ============================

A --> B1 --> B2 --> B3 --> C1 --> C2

D --> E1 --> E2
C2 -.->|Vector Search| E2
E2 --> E3 --> F1 --> F2 --> G --> H1 --> H2

%% ============================
%% CLASS ASSIGNMENTS
%% ============================

class A,B1,B2,B3 data
class C1,C2 process
class E1,E2,E3 rag
class F1,F2 llm
class G,H1,H2 ui
```

---

## 🔮 Future Work

- [ ] **Additional Data Sources** - Restaurants, hotels, activities
- [x] **Conversation History** - Multi-turn dialogue support ✅ (Chapter 05)
- [x] **Source Citations** - Verifiable answer sources ✅ (Chapter 05)
- [ ] **Map Integration** - Interactive geo-spatial visualization
- [ ] **Response Evaluation** - Quality metrics and user feedback
- [ ] **Streamlit Deployment** - Deploy to Streamlit Cloud
- [ ] **API Deployment** - RESTful API for integration

---

## 📜 License

MIT License (free to use & modify)

---

<p align="center">
  <i>Built with ❤️ using LangChain and Google Gemini Pro</i>
</p>
