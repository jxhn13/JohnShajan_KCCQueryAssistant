import pandas as pd
import re
import os
import json
from langchain.schema import Document

# File paths
RAW_CSV = "backend/kcc.csv"
CLEAN_CSV = "backend/kcc_clean.csv"
QA_JSON = "backend/kcc_qa_pairs.jsonl"
LC_DOC_JSON = "backend/kcc_documents.json"

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  
    return text

def preprocess_kcc_csv():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw KCC file not found at {RAW_CSV}")

  
    df = pd.read_csv(RAW_CSV, encoding="utf-8")


    df.rename(columns={
        "QueryText": "question",
        "KccAns": "answer"
    }, inplace=True)

 
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain 'QueryText' and 'KccAns' columns.")

    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)


    df.dropna(subset=["question", "answer"], inplace=True)
    df = df[df["question"].str.len() > 10]
    df = df[df["answer"].str.len() > 10]


    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"✅ Cleaned CSV saved to {CLEAN_CSV}")


    with open(QA_JSON, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "question": row["question"],
                "answer": row["answer"]
            }, ensure_ascii=False) + "\n")
    print(f"✅ Q&A pairs saved to {QA_JSON} (JSONL format)")


    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=f"Q: {row['question']}\nA: {row['answer']}",
            metadata={"source": "kcc.csv"}
        )
        documents.append(doc)

    with open(LC_DOC_JSON, "w", encoding="utf-8") as f:
        json.dump([doc.dict() for doc in documents], f, indent=2, ensure_ascii=False)
    print(f"✅ LangChain Documents saved to {LC_DOC_JSON}")

if __name__ == "__main__":
    preprocess_kcc_csv()
