# 📚 RAG PDF Assistant

An offline, privacy-focused Retrieval-Augmented Generation (RAG) study assistant built with Python, Streamlit, LangChain, ChromaDB, and local Ollama models.

This application converts static PDF documents—such as lecture slides, research papers, and textbook chapters—into an interactive conversational interface. It eliminates LLM hallucinations on custom documents by vectorizing uploaded text and grounding AI answers strictly within retrieved document context.

---

## 🏗️ Architecture & Workflow
[ Upload PDF ] ──► [ PyPDF Loader ] ──► [ Recursive Text Splitter ] ──► [ Nomic Embeddings ] ──► [ Chroma Vector DB ]
│
[ Local Response ] ◄── [ Ollama (Llama 3) ] ◄── [ Grounded Prompt + Top-K Chunks ] ◄──────────────┘

**Document Ingestion:** Uploaded PDFs are parsed into text and split into overlapping text chunks (`chunk_size=1000`, `chunk_overlap=200`) to preserve sentence boundaries and semantic context.
2. **Indexing:** Text chunks are transformed into dense vector embeddings using `nomic-embed-text` and indexed inside a local `ChromaDB` vector store.
3. **Retrieval:** User queries are embedded using the same vector model and matched against stored chunk vectors via similarity search to fetch the top $K=5$ relevant snippets.
4. **Grounded Generation:** The retrieved chunks are formatted into a constrained context prompt, enabling the local `llama3` model to generate factually accurate, source-grounded responses.

---

## 🛠️ Tech Stack

* **Frontend / User Interface:** [Streamlit](https://streamlit.io/)
* **RAG Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **LLM & Embeddings Engine:** [Ollama](https://ollama.com/) (`llama3` & `nomic-embed-text`)
* **PDF Parser:** `pypdf`

---

## 🚀 Getting Started

### Prerequisites

1. Python 3.10 or higher installed.
2. [Ollama](https://ollama.com/) installed and running on your machine.
3. Download the local AI models via terminal:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
