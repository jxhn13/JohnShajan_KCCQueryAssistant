🧠 KCC QA RAG - Kisan Call Center Retrieval-Augmented Generation System


This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer agricultural queries using historical Kisan Call Center (KCC) data. It combines local document-based knowledge (stored via ChromaDB) and fallback internet search (via DuckDuckGo) to ensure accurate, real-time responses


📦 Features

- 🔍 Semantic Search** over preprocessed KCC CSV dataset using Sentence-BERT
- 🧠 LLM-powered Answering** with [Ollama](https://ollama.com) (e.g., `gemma:3b`)
- 🧾 Local Knowledge Base** using `ChromaDB` as vector store
- 🌐 Fallback Web Search** via DuckDuckGo for unanswered queries
- 🔧 Fully script-based pipeline with modular functions



Create and activate a conda environment
 
  -conda create -n kcc-rag python=3.10 -y
  -conda activate kcc-rag


Install dependencies

  -pip install -r requirements.txt


If requirements.txt is not available:

   -pip install pandas chromadb sentence-transformers ollama duckduckgo-search langchain

🛠️ Setup Ollama (LLM)
ollama pull gemma3



🚀 How to Run

streamlit run app.py

This step generates:

kcc_clean.csv

kcc_qa_pairs.jsonl

kcc_documents.json


2. Load Data into ChromaDB

   This embeds cleaned questions and stores them into ChromaDB (kcc_data collection).


3. Ask a Question

   You’ll receive:

🧠 Local Answer (if relevant docs found)

🌐 Internet-based Answer (if no relevant docs found)

⚙️ Optional Flags
--k            Top-K results to retrieve (default=5)
--threshold    Relevance threshold (default=0.3)
--model        Ollama model name (default=gemma)





🧪 Dependencies
Python ≥ 3.8

Ollama (LLM runtime)

pandas

chromadb

sentence-transformers

duckduckgo-search

langchain




🧰 Future Enhancements

🌍 Translate results into regional languages

🔐 API/endpoint for external integration

📊 Vector DB upgrade to support larger KB




🙌 Acknowledgements
Ollama

LangChain

Sentence Transformers

ChromaDB

KCC (Kisan Call Center) India for real-world agri data

📞 Contact
Project by JOHN SHAJAN
📧 Email: johnshajan77@gmail.com
