# 🤖 AI Document SearchBot

**Chat with your PDFs.** Upload any set of documents and ask natural-language questions — get accurate, context-grounded answers powered by Retrieval-Augmented Generation (RAG).

🔗 **Live Demo:** [document-searchbot.onrender.com](https://document-searchbot.onrender.com)

---

## 📌 Overview

AI Document SearchBot is a full-stack **RAG (Retrieval-Augmented Generation)** application that turns static PDFs into an interactive knowledge base. Instead of manually searching through pages of text, users upload documents once and then ask questions in plain English — the app retrieves the most relevant passages and generates a grounded, hallucination-resistant answer.

It's built to be **flexible and cost-conscious**: it works with Google Gemini for best-in-class accuracy, but gracefully falls back to a **100% free, local HuggingFace embedding model** when no API key is available — making it usable by anyone, anywhere, with zero cost.

> **~85–90% QA accuracy** on evaluated document sets.

---

## ✨ Features

- 📄 **Multi-PDF ingestion** — upload and process several documents at once
- 🧠 **Dual embedding backends** — Google Gemini embeddings (`gemini-embedding-001`) *or* free local HuggingFace embeddings (`all-MiniLM-L6-v2`)
- 🔍 **Semantic search** over document content using **FAISS** vector similarity
- 💬 **Context-grounded Q&A** via **Google Gemini** (`gemini-3.1-flash-lite`) with a strict "don't hallucinate" prompt
- 📚 **Source transparency** — every answer comes with an expandable "Source Context" panel showing exactly which chunks were used
- ⚡ **Persistent vector store** — process once, query many times without re-embedding
- 🖥️ **Clean Streamlit UI** with live system status (API key detection, vector DB readiness)
- ☁️ **One-click cloud deployment** on Render

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  PDF Upload   │ --> │  Text Extraction  │ --> │   Chunking (1000    │
│ (Streamlit UI)│     │    (PyPDF2)       │     │  chars, 200 overlap)│
└──────────────┘     └──────────────────┘     └─────────┬──────────┘
                                                          │
                                                          ▼
                                            ┌──────────────────────────┐
                                            │   Embedding Generation     │
                                            │  Google Gemini  OR         │
                                            │  HuggingFace (local, free) │
                                            └─────────────┬─────────────┘
                                                          │
                                                          ▼
                                            ┌──────────────────────────┐
                                            │  FAISS Vector Store        │
                                            │  (saved locally as index)  │
                                            └─────────────┬─────────────┘
                                                          │
                        User Question                     │
                              │                            ▼
                              │              ┌──────────────────────────┐
                              └────────────> │  Similarity Search        │
                                             │  (top-k relevant chunks)  │
                                             └─────────────┬─────────────┘
                                                           │
                                                           ▼
                                             ┌──────────────────────────┐
                                             │  Gemini LLM + Prompt      │
                                             │  Chain (LangChain)         │
                                             │  → Grounded Answer         │
                                             └──────────────────────────┘
```

**Pipeline in short:** `PDF → Text → Chunks → Embeddings → FAISS Index → Similarity Search → LLM Answer`

---

## 🧰 Tech Stack

| Layer                | Technology                                                  |
|-----------------------|--------------------------------------------------------------|
| **Frontend / UI**      | [Streamlit](https://streamlit.io/)                           |
| **Orchestration**      | [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-community`) |
| **LLM (generation)**   | Google Gemini — `gemini-3.1-flash-lite`                       |
| **Embeddings**         | Google Gemini `gemini-embedding-001` **or** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store**       | [FAISS](https://github.com/facebookresearch/faiss) (CPU)      |
| **PDF Parsing**        | PyPDF2                                                        |
| **Config Management**  | python-dotenv                                                 |
| **Deployment**         | [Render](https://render.com/) (Python web service)            |
| **Language**           | Python 3.11.9                                                 |

---

## 📁 File Structure

```
document-searchbot/
├── .vscode/                # Editor workspace settings (not required to run the app)
├── venv/                   # Local Python virtual environment (git-ignored)
├── faiss_index/             # Generated at runtime — stores the FAISS vector index
│   ├── index.faiss          # Serialized FAISS vector index
│   └── index.pkl            # Pickled document store / metadata mapping for the index
├── .env                     # Local environment variables (GOOGLE_API_KEY) — git-ignored
├── .gitignore               # Excludes venv/, .env, faiss_index/, __pycache__, etc.
├── app.py                   # Main Streamlit application — UI, PDF processing, RAG pipeline
├── lab.py                   # Scratch/sandbox script used for quick local testing
├── backend_choice.txt       # Persists the last-used embedding backend ("google"/"huggingface")
├── requirements.txt         # Python dependencies
├── runtime.txt              # Pinned Python runtime version (3.11.9) for deployment
├── render.yaml              # Render.com deployment configuration (build & start commands)
├── LICENSE                  # Project license
└── README.md                # You are here
```

> **Note:** `venv/`, `.env`, and `faiss_index/` are local/runtime artifacts — they should be listed in `.gitignore` and won't (and shouldn't) appear in the GitHub repo itself. `index.faiss` and `index.pkl` are auto-generated once documents are processed.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/document-searchbot.git
cd document-searchbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```
> 💡 No API key? No problem — the app automatically falls back to the free, local HuggingFace backend.

### 5. Run the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

---

## 🧑‍💻 Usage

1. **Upload** one or more PDF files from the sidebar.
2. Click **🔄 Process Documents** — text is extracted, chunked, embedded, and indexed into FAISS.
3. Type a question in the main input box (e.g., *"What are the key findings of this report?"*).
4. Get an **AI-generated, context-grounded answer**, with a link to the exact source passages used.

---

## ☁️ Deployment

This project is pre-configured for **Render**:

- `render.yaml` defines the service (`Document_Searchbot`), Python version, build command (`pip install -r requirements.txt`), and start command (`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`).
- Simply connect the repo to Render, set the `GOOGLE_API_KEY` environment variable in the dashboard, and deploy.

It can equally be deployed on **Streamlit Community Cloud, Railway, or any Docker-compatible host** with minor tweaks.

---

## 🎯 Use Cases

- 📑 **Research assistants** — quickly extract insights from academic papers or reports
- 🏢 **Internal knowledge bases** — let teams query company policy docs, SOPs, or manuals
- ⚖️ **Legal / compliance document review** — locate relevant clauses without manual reading
- 🎓 **Study aid** — students can query textbooks or lecture notes conversationally
- 💼 **Client-facing document Q&A tools** — for consultancies handling large document sets

---

## 🛣️ Roadmap / Future Improvements

- [ ] Support for additional file types (DOCX, TXT, CSV)
- [ ] Conversation memory for multi-turn follow-up questions
- [ ] Swap FAISS for a managed vector DB (e.g., Pinecone, Qdrant) for multi-user persistence
- [ ] Add citation highlighting directly within source PDF pages
- [ ] Dockerize for consistent local/cloud parity

---

## 📄 License

This project is licensed under the terms of the `LICENSE` file included in this repository.

---

## 🙋 Author

**Mohammed Ajzel (AJ)** — AI/ML Engineer
🌐 Portfolio: [mohammed-ajzel.lovable.app](https://mohammed-ajzel.lovable.app)

If this project helped you, consider ⭐ starring the repo!
