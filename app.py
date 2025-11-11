import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from patients import build_roster_from_supabase, fuzzy_resolve
from retrieve_supabase import match_patient_chunks, match_general_documents
from query_analyzer import analyze_query_with_llm

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in .env")
    st.stop()

st.set_page_config(page_title="EHR Query Agent", layout="centered")
st.markdown("## 🏥 EHR Query Agent")
st.markdown(
    "*Ask me anything - I'll automatically detect if it's about a patient or general medical knowledge.*"
)

chat = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Load roster
with st.status("Loading patient roster...", expanded=False):
    ROSTER = build_roster_from_supabase()

if not ROSTER:
    st.error("No patients found in Supabase rag_chunks.metadata.")
    st.stop()

# Session state
if "locked_patient" not in st.session_state:
    st.session_state.locked_patient = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_disambiguation" not in st.session_state:
    st.session_state.awaiting_disambiguation = None

# Show locked patient status
if st.session_state.locked_patient:
    p = st.session_state.locked_patient
    col1, col2 = st.columns([4, 1])
    with col1:
        st.success(
            f"🔒 **Active Patient:** {p['first_name']} {p['last_name']} (ID: `{p['patient_id']}`, DOB: {p['dob']})"
        )
    with col2:
        if st.button("Clear"):
            st.session_state.locked_patient = None
            st.session_state.messages.append({
                "role":
                "assistant",
                "content":
                "Patient context cleared. You can now ask general questions or mention a different patient."
            })
            st.rerun()

st.markdown("---")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources") and msg["role"] == "assistant":
            with st.expander("📚 Sources"):
                st.json(msg["sources"])

# Chat input
prompt = st.chat_input(
    "Ask anything... (e.g., 'What's the protocol for fever?' or 'Show me Alex's latest MRI')"
)

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check if awaiting disambiguation response
    if st.session_state.awaiting_disambiguation:
        # User is responding to disambiguation
        user_choice = prompt.strip().lower()
        candidates = st.session_state.awaiting_disambiguation

        # Try to match user's choice
        selected = None
        for idx, candidate in enumerate(candidates):
            # Check if user typed a number (1, 2, etc.)
            if user_choice == str(idx + 1):
                selected = candidate
                break
            # Check if user typed the patient ID
            if user_choice == candidate['patient_id'].lower():
                selected = candidate
                break
            # Check if user typed part of the name
            full_name = f"{candidate['first_name']} {candidate['last_name']}".lower(
            )
            if user_choice in full_name or full_name in user_choice:
                selected = candidate
                break

        if selected:
            st.session_state.locked_patient = selected
            st.session_state.awaiting_disambiguation = None
            response = f"✅ Locked to patient **{selected['first_name']} {selected['last_name']}** (DOB: {selected['dob']}). What would you like to know?"
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
        else:
            response = "I couldn't match your selection. Please try again by typing the number, patient ID, or name."
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()

    # Use LLM-based routing
    with st.spinner("Analyzing query..."):
        from query_analyzer import analyze_query_with_llm
        analysis = analyze_query_with_llm(prompt, ROSTER,
                                          st.session_state.locked_patient)

    # Handle different routing scenarios
    if analysis["resolved_patient"]:
        # Unique patient found - lock to it
        patient = analysis["resolved_patient"]

        # Only update lock if it's a different patient
        if not st.session_state.locked_patient or st.session_state.locked_patient[
                'patient_id'] != patient['patient_id']:
            st.session_state.locked_patient = patient

        # Get patient context
        with st.spinner(
                f"Retrieving records for {patient['first_name']} {patient['last_name']}..."
        ):
            hits = match_patient_chunks(prompt, patient["patient_id"], k=6)

        context = [h["content"] for h in hits]
        sources = [h["metadata"] for h in hits]
        system = f"You are a clinical assistant. Use ONLY the retrieved patient context for {patient['first_name']} {patient['last_name']}."

    elif analysis["candidates"]:
        # Multiple patients found - ask for clarification
        st.session_state.awaiting_disambiguation = analysis["candidates"]
        response = f"I found multiple patients matching '{analysis['patient_reference']}'. Did you mean:\n\n"
        for idx, candidate in enumerate(analysis["candidates"]):
            response += f"{idx + 1}. **{candidate['first_name']} {candidate['last_name']}** (DOB: {candidate['dob']}, ID: `{candidate['patient_id']}`)\n"
        response += "\nPlease type the number, patient ID, or full name to select."
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        st.rerun()

    elif analysis["intent"] == "patient_specific_use_locked":
        # Using locked patient context
        patient = st.session_state.locked_patient
        with st.spinner(
                f"Retrieving records for {patient['first_name']} {patient['last_name']}..."
        ):
            hits = match_patient_chunks(prompt, patient["patient_id"], k=6)

        context = [h["content"] for h in hits]
        sources = [h["metadata"] for h in hits]
        system = f"You are a clinical assistant. Use ONLY the retrieved patient context for {patient['first_name']} {patient['last_name']}."

    elif analysis["intent"] == "patient_specific_no_context":
        # Patient-specific query but no patient locked or found
        response = "It seems you're asking about a specific patient, but I need to know which patient. Could you please mention the patient's name or ID?"
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        st.rerun()

    elif analysis["intent"] == "patient_specific_not_found":
        # Patient name detected but not in roster
        response = f"I couldn't find a patient matching '{analysis['patient_reference']}' in the system. Please check the name or ID and try again."
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        st.rerun()

    else:
        # General medical query
        with st.spinner("Searching medical knowledge base..."):
            hits = match_general_documents(prompt, k=6)

        context = [h["content"] for h in hits]
        sources = [h["metadata"] for h in hits]
        system = "You are a medical knowledge assistant. Use ONLY the retrieved document context to answer questions."

    # Build conversation with memory (last 7 messages)
    ctx_block = "\n---\n".join(context[:8]) if context else "(no context)"
    messages = [{"role": "system", "content": system}]

    # Add last 7 messages from history
    recent_history = st.session_state.messages[-8:-1] if len(
        st.session_state.messages) > 1 else []
    for msg in recent_history[-7:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current query with context
    messages.append({
        "role":
        "user",
        "content":
        f"CONTEXT:\n{ctx_block}\n\nQUESTION: {prompt}\nAnswer:"
    })

    # Stream response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        with st.spinner("Thinking..."):
            for chunk in chat.stream(messages):
                streamed += chunk.content or ""
                placeholder.markdown(streamed)

    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": streamed,
        "sources": sources
    })
    st.rerun()
