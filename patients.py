from typing import Dict, Any, List, Tuple, Optional
from rapidfuzz import process, fuzz
from supabase_client import get_supabase

ALIASES = {
    "patient_id": ["patient_id","Patient_Id","PatientID"],
    "first_name": ["first_name","First_Name"],
    "last_name": ["last_name","Last_Name"],
    "dob": ["dob","Date_of_birth","DOB"]
}

def mget(md: Dict[str, Any], key: str) -> str:
    for k in ALIASES.get(key, [key]):
        if k in md:
            return str(md[k])
    return ""

def build_roster_from_supabase(limit: int = 20000) -> List[Dict[str, str]]:
    sb = get_supabase()
    res = sb.table("rag_chunks").select("metadata").limit(limit).execute()
    by_pid = {}
    for row in (res.data or []):
        md = row.get("metadata") or {}
        pid = mget(md, "patient_id")
        if not pid: continue
        if pid not in by_pid:
            by_pid[pid] = {
                "patient_id": pid,
                "first_name": mget(md, "first_name"),
                "last_name": mget(md, "last_name"),
                "dob": mget(md, "dob")
            }
    return list(by_pid.values())

def fuzzy_resolve(roster: List[Dict[str,str]], q: str) -> Tuple[Optional[Dict[str,str]], List[Dict[str,str]], str]:
    q = (q or "").strip()
    if not q: return None, [], "none"
    id_hits = [r for r in roster if q.lower() in r["patient_id"].lower()]
    if len(id_hits)==1: return id_hits[0], [], "by_id"
    if len(id_hits)>1: return None, id_hits, "ambiguous"
    names = [f"{r['first_name']} {r['last_name']}".strip() for r in roster]
    if names:
        best = process.extractOne(q, names, scorer=fuzz.WRatio)
        if best and best[1]>=80: return roster[best[2]], [], "by_name"
        cands = process.extract(q, names, scorer=fuzz.WRatio, limit=5)
        cands = [roster[idx] for name,score,idx in cands if score>=60]
        if cands: return None, cands, "ambiguous"
    return None, [], "none"
