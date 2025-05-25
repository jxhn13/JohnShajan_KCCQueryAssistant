import os
import json
import csv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import ollama
from backend.preprocess import preprocess_kcc_csv, CLEAN_CSV
from backend.livesearch import live_internet_search_duckduckgo

# -------- Config --------

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "kcc_data"
JSONL_PATH = "backend/kcc_qa_pairs.jsonl"
BATCH_SIZE = 16
OLLAMA_MODEL = "gemma3"
RELEVANCE_THRESHOLD = 0.3
TOP_K = 5



SentenceTransformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME
)

chroma_client = chromadb.Client()
ollama_client = ollama.Client()

try:
    collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=SentenceTransformer_ef)



def get_embeddings_batch(texts):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    return embeddings

# -------- Data Loading and Indexing --------

def load_kcc_data(csv_path=CLEAN_CSV, force_reload=False):
    global collection

    if not force_reload and os.path.exists(JSONL_PATH):
        try:
            collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
            print(f"[INFO] Chroma collection '{CHROMA_COLLECTION_NAME}' loaded from existing index.")
            return
        except Exception:
            print("[WARN] Could not load existing Chroma collection. Rebuilding...")

    preprocess_kcc_csv()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    documents = []
    ids = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        question_idx = header.index("question")
        answer_idx = header.index("answer")

        for i, row in enumerate(reader):
            if len(row) <= max(question_idx, answer_idx):
                continue
            question = row[question_idx].strip()
            answer = row[answer_idx].strip()
            documents.append(f"Q: {question}\nA: {answer}")
            ids.append(f"kcc_{i}")

    embeddings = get_embeddings_batch(documents)

    try:
        existing_ids = collection.get().get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)
    except Exception:
        collection = chroma_client.create_collection(name=CHROMA_COLLECTION_NAME)

    collection.add(documents=documents, embeddings=embeddings, ids=ids)

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents):
            json.dump({
                "page_content": doc,
                "metadata": {"source": "kcc", "id": i}
            }, f)
            f.write("\n")

    print(f"[INFO] Loaded {len(documents)} Q&A pairs into ChromaDB.")

# -------- Query and Answer Generation --------

def generate_answer(query: str, top_k: int = TOP_K, relevance_threshold: float = RELEVANCE_THRESHOLD) -> dict:
    results = collection.query(
        query_texts=[query],  
        n_results=top_k,
        include=["documents", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    print(f"[DEBUG] Retrieved documents: {documents}")
    print(f"[DEBUG] Distances: {distances}")

    relevant_docs = [
        doc for doc, dist in zip(documents, distances)
        if dist <= (1 - relevance_threshold)
    ]

    print(f"[DEBUG] Relevant docs: {relevant_docs}")

    local_answer = None
    internet_answer = None

    if relevant_docs:
        context = "\n\n".join(relevant_docs)
        prompt = (
            f"You are an AI assistant for the Kisan Call Center agricultural dataset.\n"
            f"Use the following context to answer the user's question in the same language as the context or question.\n\n"
            f"Then, provide the English translation of the answer below it.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer in local language and then its English translation:"
        )

        try:
            response = ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            message = response.get("message", None)
            if message and hasattr(message, "content"):
                local_answer = message.content.strip()
        except Exception as e:
            print(f"[ERROR] Ollama chat failed for local data: {e}")
            local_answer = "⚠️ Error generating answer from the local KCC dataset."

    else:
        print("[INFO] No relevant local data found. Performing internet search...")
        search_results = live_internet_search_duckduckgo(query)

        if search_results:
            snippets = "\n".join([res["snippet"] for res in search_results if "snippet" in res][:3])
            prompt = (
                f"⚠️ No relevant information was found in the local KCC dataset.\n\n"
                f"To assist the user, a live internet search was performed.\n\n"
                f"Based on the following search results, generate a helpful answer:\n\n"
                f"{snippets}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )

            try:
                response = ollama_client.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}]
                )
                message = response.get("message", {})
                internet_answer = message.get("content", "").strip()
            except Exception as e:
                internet_answer = f"⚠️ Error using live internet data with Ollama: {e}"
        else:
            internet_answer = "⚠️ No relevant context found in local data or internet search. Please try rephrasing your query."

    return {
        "local_answer": local_answer,
        "internet_answer": internet_answer
    }
