# ingest_sample.py
import os
from dotenv import load_dotenv
from supabase import create_client
from langchain_openai import OpenAIEmbeddings

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SERVICE_SUPABASESERVICE_KEY"]  # use service role locally for writes
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
emb = OpenAIEmbeddings(model=EMBED_MODEL)

# two tiny example chunks for patient IVF001
chunks = [
    ("Medication list: Letrozole 2.5 mg daily; Folic acid 5 mg.", {
        "patient_id":"IVF001","First_Name":"Priya","Last_Name":"Sharma","Date_of_birth":"1988-03-15",
        "doc_id":"meds_2025.txt"
    }),
    ("Imaging: MRI pelvis 2025-09-14 shows adenomyosis; no adnexal mass.", {
        "patient_id":"IVF001","First_Name":"Priya","Last_Name":"Sharma","Date_of_birth":"1988-03-15",
        "doc_id":"imaging_2025.txt"
    }),
]

texts = [c[0] for c in chunks]
vecs  = emb.embed_documents(texts)  # 1536-d each

rows = []
for (text, md), v in zip(chunks, vecs):
    rows.append({"content": text, "metadata": md, "embedding": v})

resp = sb.table("rag_chunks").insert(rows).execute()
print("Inserted:", len(resp.data or []))
