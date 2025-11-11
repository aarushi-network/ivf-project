from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from patients import fuzzy_resolve
import os

# Use a smaller, faster model for routing
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
router_llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0)

# Define the routing prompt
ROUTING_PROMPT = ChatPromptTemplate.from_messages([(
    "system",
    """You are a query routing assistant for a medical EHR system. Analyze the user's query and determine:

1. **intent**: Is this a "patient_specific" query (asking about a specific patient's records) or a "general" query (asking about general medical knowledge)?

2. **patient_reference**: If patient-specific, extract any patient name or ID mentioned. Return null if none found.

3. **confidence**: Your confidence level (0.0 to 1.0) in the intent classification.

Examples:
- "What's the protocol for postoperative fever?" → intent: "general", patient_reference: null
- "Show me Alex's latest MRI results" → intent: "patient_specific", patient_reference: "Alex"
- "What medications is Sarah Johnson on?" → intent: "patient_specific", patient_reference: "Sarah Johnson"
- "Patient IVF001's lab results" → intent: "patient_specific", patient_reference: "IVF001"
- "What are his current vitals?" → intent: "patient_specific", patient_reference: null (context needed)
- "Explain hypertension treatment guidelines" → intent: "general", patient_reference: null

Return a JSON object with these fields: intent, patient_reference, confidence"""
), ("user", "{query}")])


def analyze_query_with_llm(
        query: str,
        roster: List[Dict[str, str]],
        locked_patient: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Use an LLM to analyze the query and route it appropriately.

    Args:
        query: The user's natural language query
        roster: List of all patients in the system
        locked_patient: Currently locked patient (if any)

    Returns:
        Dict with: intent, patient_reference, resolved_patient, candidates, confidence
    """
    # Get LLM routing decision
    chain = ROUTING_PROMPT | router_llm | JsonOutputParser()

    try:
        routing_decision = chain.invoke({"query": query})
    except Exception as e:
        # Fallback to general if LLM fails
        routing_decision = {
            "intent": "general",
            "patient_reference": None,
            "confidence": 0.5
        }

    result = {
        "intent": routing_decision.get("intent", "general"),
        "patient_reference": routing_decision.get("patient_reference"),
        "confidence": routing_decision.get("confidence", 0.5),
        "resolved_patient": None,
        "candidates": []
    }

    # If patient-specific intent
    if result["intent"] == "patient_specific":
        patient_ref = result["patient_reference"]

        if patient_ref:
            # Try to resolve the patient using fuzzy matching
            resolved, candidates, reason = fuzzy_resolve(roster, patient_ref)

            if resolved:
                result["resolved_patient"] = resolved
            elif candidates:
                result["candidates"] = candidates
            else:
                # LLM detected patient intent but we can't find the patient
                result["intent"] = "patient_specific_not_found"
        else:
            # LLM says patient-specific but no name extracted
            # Check if we have a locked patient to use
            if locked_patient:
                result["resolved_patient"] = locked_patient
                result["intent"] = "patient_specific_use_locked"
            else:
                result["intent"] = "patient_specific_no_context"

    return result
