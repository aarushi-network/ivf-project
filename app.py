# app.py (streaming chat + Supabase RAG) — no duplicate assistant message
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from patients import build_roster_from_supabase, fuzzy_resolve
from retrieve_supabase import match_patient_chunks

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in .env")
    st.stop()

st.set_page_config(page_title="EHR Query Agent", layout="centered")
st.markdown("## EHR Query Agent")

chat = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Load roster
with st.status("Patient data pre-loaded...", expanded=False):
    ROSTER = build_roster_from_supabase()

if not ROSTER:
    st.error("No patients found in Supabase rag_chunks.metadata.")
    st.stop()

# Session state
if "locked" not in st.session_state:
    st.session_state.locked = None
if "pending" not in st.session_state:
    st.session_state.pending = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role","content","sources"?}]

# ───────────────── Patient selection UI ─────────────────
st.markdown("### Select & Lock Patient")


def resolve_patient():
    q = st.session_state.patient_query
    if q:
        resolved, candidates, reason = fuzzy_resolve(ROSTER, q)
        if resolved:
            st.session_state.pending = resolved
            st.session_state.resolve_message = (
                "info",
                f"Detected: **{resolved['first_name']} {resolved['last_name']}** (`{resolved['patient_id']}`)"
            )
        elif candidates:
            st.session_state.pending = None
            st.session_state.resolve_message = ("warning",
                                                "Multiple matches found")
            st.session_state.candidates = candidates
        else:
            st.session_state.pending = None
            st.session_state.resolve_message = ("error",
                                                "No match found. Try again.")
            st.session_state.candidates = []
    else:
        st.session_state.pending = None
        st.session_state.resolve_message = None
        st.session_state.candidates = []


# Initialize candidates in session state if not present
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "resolve_message" not in st.session_state:
    st.session_state.resolve_message = None

q = st.text_input("Patient ID or Name",
                  key="patient_query",
                  on_change=resolve_patient)

# Display resolve message if exists
if st.session_state.resolve_message:
    msg_type, msg_text = st.session_state.resolve_message
    if msg_type == "info":
        st.info(msg_text)
    elif msg_type == "warning":
        st.warning(msg_text)
        for c in st.session_state.candidates:
            st.write(
                f"- `{c['patient_id']}` — {c['first_name']} {c['last_name']} (DOB {c['dob']})"
            )
    elif msg_type == "error":
        st.error(msg_text)

if st.session_state.pending:
    p = st.session_state.pending
    with st.form(key="dob_form"):
        dob_in = st.text_input("Confirm DOB (YYYY-MM-DD)", key="dob_confirm")
        submit_button = st.form_submit_button("Confirm patient")
        if submit_button:
            if dob_in.strip() == (p["dob"] or "").strip():
                st.session_state.locked = p
                st.session_state.pending = None
                st.session_state.messages.clear()
                st.session_state.resolve_message = None
                st.success(
                    f"Locked to patient: **{p['first_name']} {p['last_name']}** (`{p['patient_id']}`)"
                )
                st.rerun()
            else:
                st.error("DOB does not match.")

if st.session_state.locked:
    lp = st.session_state.locked
    st.success(
        f"**Locked:** {lp['first_name']} {lp['last_name']} (`{lp['patient_id']}`) — DOB: {lp['dob']}"
    )
else:
    st.info("No patient locked yet.")

# ───────────────── Chat UI ─────────────────
st.markdown("### Chat")

# 1) Render existing history FIRST (prevents duplicate of the streamed reply)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources") and msg["role"] == "assistant":
            with st.expander("Sources"):
                st.json(msg["sources"])

# 2) Collect new input
prompt = st.chat_input(
    "Ask about this patient (e.g., 'What medications is this patient on?')")

# 3) Handle the prompt after rendering history
if prompt:
    # append user msg
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not st.session_state.locked:
        st.session_state.messages.append({
            "role":
            "assistant",
            "content":
            "Please lock a patient first."
        })
        st.rerun()

    # RAG for locked patient
    pid = st.session_state.locked["patient_id"]
    with st.spinner("Retrieving context..."):
        hits = match_patient_chunks(prompt, pid, k=6)

    context = [h["content"] for h in hits]
    sources = [h["metadata"] for h in hits]
    system = "You are a clinical assistant. Use ONLY the retrieved patient context."
    ctx_block = "\n---\n".join(context[:8]) if context else "(no context)"
    messages = [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{ctx_block}\n\nQUESTION: {prompt}\nAnswer:"
        },
    ]

    # 4) Stream the assistant reply LIVE (not added to history yet)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        with st.spinner("Thinking..."):
            for chunk in chat.stream(messages):
                streamed += chunk.content or ""
                placeholder.markdown(streamed)

    # 5) Now persist the assistant message and refresh the app
    st.session_state.messages.append({
        "role": "assistant",
        "content": streamed,
        # "sources": sources
    })
    st.rerun()
