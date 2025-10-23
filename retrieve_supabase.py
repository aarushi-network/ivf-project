from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from supabase_client import get_supabase

load_dotenv()
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
emb = OpenAIEmbeddings(model=EMBED_MODEL)

def match_patient_chunks(query: str, patient_id: str, k: int=6) -> List[Dict[str,Any]]:
    """
    Calls SQL function:
      match_patient_chunks(query_embedding vector, match_count int, p_patient_id text)
    Returns rows: {id, content, metadata, similarity}
    """
    sb = get_supabase()
    qvec = emb.embed_query(query)
    res = sb.rpc("match_patient_chunks_arr", {
        "query_embedding": qvec,
        "match_count": k,
        "p_patient_id": patient_id
    }).execute()
    return res.data or []
