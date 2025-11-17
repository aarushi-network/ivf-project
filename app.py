import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from patients import build_roster_from_supabase, fuzzy_resolve
from retrieve_supabase import match_patient_chunks, match_general_documents
from query_analyzer import analyze_query_with_slm

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
if "active_patients" not in st.session_state:
    st.session_state.active_patients = []  # For multi-patient queries
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_disambiguation" not in st.session_state:
    st.session_state.awaiting_disambiguation = None

# Show active patient(s) status
if st.session_state.active_patients and len(st.session_state.active_patients) > 0:
    # Multi-patient active
    col1, col2 = st.columns([4, 1])
    with col1:
        if len(st.session_state.active_patients) == 1:
            # Single patient
            p = st.session_state.active_patients[0]
            st.success(
                f"🔒 **Active Patient:** {p['first_name']} {p['last_name']} (ID: `{p['patient_id']}`, DOB: {p['dob']})"
            )
        else:
            # Multiple patients
            patients_list = ", ".join([
                f"{p['first_name']} {p['last_name']} (ID: `{p['patient_id']}`)"
                for p in st.session_state.active_patients
            ])
            st.success(
                f"🔒 **Active Patients:** {patients_list}"
            )
    with col2:
        if st.button("Clear"):
            st.session_state.active_patients = []
            st.session_state.locked_patient = None
            st.session_state.messages.append({
                "role":
                "assistant",
                "content":
                "Patient context cleared. You can now ask general questions or mention different patients."
            })
            st.rerun()
elif st.session_state.locked_patient:
    # Legacy single patient (for backward compatibility)
    p = st.session_state.locked_patient
    col1, col2 = st.columns([4, 1])
    with col1:
        st.success(
            f"🔒 **Active Patient:** {p['first_name']} {p['last_name']} (ID: `{p['patient_id']}`, DOB: {p['dob']})"
        )
    with col2:
        if st.button("Clear"):
            st.session_state.locked_patient = None
            st.session_state.active_patients = []
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
        # if msg.get("sources") and msg["role"] == "assistant":
            # with st.expander("📚 Sources"):
            #     st.json(msg["sources"])

# Chat input
prompt = st.chat_input(
    "Ask anything..."
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
        from query_analyzer import analyze_query_with_slm
        analysis = analyze_query_with_slm(prompt, ROSTER,
                                          st.session_state.locked_patient)

    # Handle different routing scenarios
    if analysis.get("resolved_patients") and len(analysis["resolved_patients"]) >= 2:
        # Multi-patient query - retrieve context for all patients
        patients = analysis["resolved_patients"]
        
        # Set active patients for UI display
        st.session_state.active_patients = patients
        st.session_state.locked_patient = None  # Clear single patient lock
        
        all_context = []
        all_sources = []
        
        # For multi-patient queries, retrieve more chunks per patient to ensure we get relevant data
        # Create a patient-agnostic query to avoid bias towards specific patient names
        query_lower = prompt.lower()
        
        # Extract the specific attribute being asked about (height, weight, etc.) without patient names
        # This ensures equal retrieval quality for all patients
        if "height" in query_lower:
            # Use generic height query without patient names for better matching
            retrieval_query = "height measurement cm"
        elif "weight" in query_lower:
            retrieval_query = "weight measurement kg"
        elif "blood pressure" in query_lower or "bp" in query_lower:
            retrieval_query = "blood pressure measurement"
        elif "temperature" in query_lower or "temp" in query_lower:
            retrieval_query = "temperature measurement"
        elif "bmi" in query_lower:
            retrieval_query = "bmi body mass index"
        elif any(term in query_lower for term in ["compare", "comparison", "difference"]):
            # For general comparison queries, extract what's being compared
            # Try to find measurement terms
            if any(term in query_lower for term in ["vital", "lab", "test", "result"]):
                retrieval_query = "patient measurement data"
            else:
                retrieval_query = "patient information data"
        else:
            # Use the original query but remove patient names for better matching
            # Keep only the attribute/measurement terms
            retrieval_query = prompt
        
        # Store chunks per patient first, then interleave them
        patient_chunks = {}
        with st.spinner(f"Retrieving records for {len(patients)} patients..."):
            for patient in patients:
                patient_name = f"{patient['first_name']} {patient['last_name']}"
                patient_hits = []
                seen_chunk_ids = set()
                
                # Strategy 1: Use patient-agnostic attribute query (retrieve more chunks)
                hits1 = match_patient_chunks(retrieval_query, patient["patient_id"], k=20)
                for h in hits1:
                    chunk_id = h.get("id") or h.get("content", "")[:50]
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        patient_hits.append((h, patient_name))
                
                # Strategy 2: Also try with patient name included
                patient_query = f"{patient['first_name']} {retrieval_query}"
                hits2 = match_patient_chunks(patient_query, patient["patient_id"], k=15)
                for h in hits2:
                    chunk_id = h.get("id") or h.get("content", "")[:50]
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        patient_hits.append((h, patient_name))
                
                # Strategy 3: General patient data as fallback (retrieve more)
                general_hits = match_patient_chunks("patient information data", patient["patient_id"], k=15)
                for h in general_hits:
                    chunk_id = h.get("id") or h.get("content", "")[:50]
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        patient_hits.append((h, patient_name))
                
                # Strategy 4: If asking about height, also try very specific height queries
                if "height" in prompt.lower():
                    height_queries = [
                        f"{patient['first_name']} height",
                        "height cm",
                        f"height {patient['first_name']}"
                    ]
                    for hq in height_queries:
                        height_hits = match_patient_chunks(hq, patient["patient_id"], k=10)
                        for h in height_hits:
                            chunk_id = h.get("id") or h.get("content", "")[:50]
                            if chunk_id not in seen_chunk_ids:
                                seen_chunk_ids.add(chunk_id)
                                patient_hits.append((h, patient_name))
                
                patient_chunks[patient_name] = patient_hits
        
        # Verify we have chunks for all patients - if not, do emergency retrieval
        for patient in patients:
            patient_name = f"{patient['first_name']} {patient['last_name']}"
            if patient_name not in patient_chunks or len(patient_chunks[patient_name]) == 0:
                # Emergency: try multiple very broad queries
                emergency_queries = [
                    patient['first_name'],
                    patient['patient_id'],
                    f"{patient['first_name']} {patient['last_name']}",
                    "patient data"
                ]
                emergency_hits = []
                for eq in emergency_queries:
                    hits = match_patient_chunks(eq, patient["patient_id"], k=15)
                    emergency_hits.extend(hits)
                    if len(emergency_hits) >= 15:
                        break
                # Deduplicate
                seen = set()
                for h in emergency_hits:
                    chunk_id = h.get("id") or h.get("content", "")[:50]
                    if chunk_id not in seen:
                        seen.add(chunk_id)
                        if patient_name not in patient_chunks:
                            patient_chunks[patient_name] = []
                        patient_chunks[patient_name].append((h, patient_name))
        
        # Interleave chunks from all patients to ensure equal visibility
        # This prevents the LLM from focusing only on the first patient's data
        max_chunks = max(len(chunks) for chunks in patient_chunks.values()) if patient_chunks else 0
        interleaved_context = []
        interleaved_sources = []
        
        # Create a summary header showing what we have for each patient
        summary_parts = []
        for patient in patients:
            patient_name = f"{patient['first_name']} {patient['last_name']}"
            chunk_count = len(patient_chunks.get(patient_name, []))
            summary_parts.append(f"{patient_name}: {chunk_count} data chunks")
        summary = f"CONTEXT SUMMARY: {' | '.join(summary_parts)}\n\n"
        interleaved_context.append(summary)
        
        # Interleave: take one chunk from each patient in round-robin fashion
        for i in range(max_chunks):
            for patient in patients:
                patient_name = f"{patient['first_name']} {patient['last_name']}"
                chunks = patient_chunks.get(patient_name, [])
                if i < len(chunks):
                    h, pname = chunks[i]
                    interleaved_context.append(f"[{pname}]: {h['content']}")
                    interleaved_sources.append(h.get("metadata"))
        
        context = interleaved_context
        sources = interleaved_sources
        
        # Build explicit patient list for the prompt
        patient_list = "\n".join([f"- {p['first_name']} {p['last_name']} (ID: {p['patient_id']})" for p in patients])
        patient_names = ", ".join([f"{p['first_name']} {p['last_name']}" for p in patients])
        
        # Extract what attribute is being asked about
        attribute = "the requested information"
        if "height" in prompt.lower():
            attribute = "height"
        elif "weight" in prompt.lower():
            attribute = "weight"
        elif "blood pressure" in prompt.lower() or "bp" in prompt.lower():
            attribute = "blood pressure"
        
        system = f"""You are a clinical assistant. You have been asked to compare information for these patients:
{patient_list}

The context below contains interleaved data chunks from ALL patients. Each chunk is labeled with [Patient Name]: to identify which patient it belongs to.

CRITICAL INSTRUCTIONS:
1. You MUST extract {attribute} for EACH of these patients: {patient_names}
2. Search through ALL chunks in the context - data is interleaved, so look for chunks labeled with each patient's name
3. For each patient, find their {attribute} value in the chunks labeled with their name
4. If you find {attribute} for a patient, use it in your comparison
5. If you cannot find {attribute} for a specific patient after searching ALL their labeled chunks, then and only then state that it's missing
6. Present your answer comparing ALL patients mentioned: {patient_names}

Remember: The context summary at the top shows how many chunks you have for each patient. Use ALL of them."""
    
    elif analysis["resolved_patient"]:
        # Unique patient found - lock to it
        patient = analysis["resolved_patient"]

        # Only update lock if it's a different patient
        if not st.session_state.locked_patient or st.session_state.locked_patient[
                'patient_id'] != patient['patient_id']:
            st.session_state.locked_patient = patient
            # Set as single active patient
            st.session_state.active_patients = [patient]

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
        # Ensure active_patients is set
        if not st.session_state.active_patients:
            st.session_state.active_patients = [patient]
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
        unresolved = analysis.get("unresolved_refs", [])
        if unresolved:
            unresolved_str = ", ".join(unresolved)
            response = f"I couldn't find patient(s) matching '{unresolved_str}' in the system. Please check the names or IDs and try again."
        else:
            response = f"I couldn't find a patient matching '{analysis.get('patient_reference', 'the mentioned patient')}' in the system. Please check the name or ID and try again."
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
    # For multi-patient queries, allow more context chunks (at least 15 per patient)
    # For single patient, use reasonable limit
    if analysis.get("resolved_patients") and len(analysis.get("resolved_patients", [])) >= 2:
        num_patients = len(analysis["resolved_patients"])
        # Allow at least 15 chunks per patient to ensure all data is included
        max_context_chunks = max(50, num_patients * 15)
        ctx_block = "\n---\n".join(context[:max_context_chunks]) if context else "(no context)"
    else:
        # Single patient or general query - use reasonable limit
        ctx_block = "\n---\n".join(context[:20]) if context else "(no context)"
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
        # "sources": sources
    })
    st.rerun()
