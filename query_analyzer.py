from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from patients import fuzzy_resolve
import os
import re

# Use an SLM (Small Language Model) for routing - faster and cheaper
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-3.5-turbo")
router_llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0)

# Define the routing prompt
ROUTING_PROMPT = ChatPromptTemplate.from_messages([(
    "system",
    """You are a query routing assistant for a medical EHR system. Analyze the user's query and determine:

1. **intent**: Is this a "patient_specific" query (asking about a specific patient's records), "multi_patient" query (comparing or asking about multiple patients), or a "general" query (asking about general medical knowledge)?

2. **patient_reference**: If patient-specific, extract the patient name or ID mentioned. Return null if none found. For multi-patient queries, this should be the first patient mentioned.

3. **patient_references**: For multi-patient queries, extract ALL patient names or IDs mentioned as a list. CRITICAL: Extract EVERY patient name mentioned, including the last one. Look for patterns like:
   - "X, Y, and Z" → ["X", "Y", "Z"]
   - "X's, Y's, and Z's" → ["X", "Y", "Z"]
   - "X and Y" → ["X", "Y"]
   - "compare X with Y and Z" → ["X", "Y", "Z"]
   Return empty list if not multi-patient.

4. **confidence**: Your confidence level (0.0 to 1.0) in the intent classification.

Examples:
- "What's the protocol for postoperative fever?" → intent: "general", patient_reference: null, patient_references: []
- "Show me Alex's latest MRI results" → intent: "patient_specific", patient_reference: "Alex", patient_references: []
- "Compare Priya's height with Meera's height" → intent: "multi_patient", patient_reference: "Priya", patient_references: ["Priya", "Meera"]
- "Compare Priya's, Meera's and Sneha's height" → intent: "multi_patient", patient_reference: "Priya", patient_references: ["Priya", "Meera", "Sneha"]
- "Show me heights of Priya, Meera, and Sneha" → intent: "multi_patient", patient_reference: "Priya", patient_references: ["Priya", "Meera", "Sneha"]
- "What medications is Sarah Johnson on?" → intent: "patient_specific", patient_reference: "Sarah Johnson", patient_references: []
- "Patient IVF001's lab results" → intent: "patient_specific", patient_reference: "IVF001", patient_references: []
- "What are his current vitals?" → intent: "patient_specific", patient_reference: null, patient_references: []
- "Explain hypertension treatment guidelines" → intent: "general", patient_reference: null, patient_references: []

IMPORTANT: When extracting patient_references, make sure to include ALL patients mentioned, especially the last one after "and". Return a JSON object with these fields: intent, patient_reference, patient_references, confidence"""
), ("user", "{query}")])


def analyze_query_with_slm(
        query: str,
        roster: List[Dict[str, str]],
        locked_patient: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Use an SLM (Small Language Model) to analyze the query and route it appropriately.

    Args:
        query: The user's natural language query
        roster: List of all patients in the system
        locked_patient: Currently locked patient (if any)

    Returns:
        Dict with: intent, patient_reference, resolved_patient, candidates, confidence
    """
    # Get SLM routing decision
    chain = ROUTING_PROMPT | router_llm | JsonOutputParser()

    try:
        routing_decision = chain.invoke({"query": query})
    except Exception as e:
        # Fallback to general if SLM fails
        routing_decision = {
            "intent": "general",
            "patient_reference": None,
            "confidence": 0.5
        }

    result = {
        "intent": routing_decision.get("intent", "general"),
        "patient_reference": routing_decision.get("patient_reference"),
        "patient_references": routing_decision.get("patient_references", []),
        "confidence": routing_decision.get("confidence", 0.5),
        "resolved_patient": None,
        "resolved_patients": [],  # For multi-patient queries
        "candidates": []
    }

    # Handle multi-patient queries
    if result["intent"] == "multi_patient":
        patient_refs = result.get("patient_references", [])
        
        # Fallback: If SLM didn't extract enough references, try to extract from query directly
        # Look for patterns like "X, Y, and Z" or "X's, Y's, and Z's"
        if len(patient_refs) < 2:
            # Try to extract patient names from the query
            # Pattern: names separated by commas and/or "and"
            name_pattern = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:'s)?(?:\s*,\s*|\s+and\s+)"
            matches = re.findall(name_pattern, query, re.IGNORECASE)
            if matches:
                # Also check for last name after "and"
                and_pattern = r"and\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:'s)?"
                and_matches = re.findall(and_pattern, query, re.IGNORECASE)
                if and_matches:
                    matches.extend(and_matches)
                # Add extracted names to patient_refs if not already there
                for match in matches:
                    if match and match not in patient_refs:
                        patient_refs.append(match)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_refs = []
        for ref in patient_refs:
            ref_lower = ref.lower().strip() if ref else ""
            if ref_lower and ref_lower not in seen:
                seen.add(ref_lower)
                unique_refs.append(ref)
        patient_refs = unique_refs
        
        if len(patient_refs) >= 2:
            resolved_patients = []
            unresolved_refs = []
            for ref in patient_refs:
                if ref:  # Only process non-empty references
                    resolved, candidates, reason = fuzzy_resolve(roster, ref)
                    if resolved:
                        # Check if we already have this patient (avoid duplicates)
                        if not any(p['patient_id'] == resolved['patient_id'] for p in resolved_patients):
                            resolved_patients.append(resolved)
                    else:
                        unresolved_refs.append(ref)
            
            if len(resolved_patients) >= 2:
                result["resolved_patients"] = resolved_patients
                result["intent"] = "multi_patient"
                # Store unresolved refs for debugging/error messages
                if unresolved_refs:
                    result["unresolved_refs"] = unresolved_refs
            elif len(resolved_patients) == 1 and len(patient_refs) >= 2:
                # Only one patient resolved but multiple were mentioned
                result["intent"] = "patient_specific_not_found"
                result["unresolved_refs"] = unresolved_refs
            else:
                # Couldn't resolve enough patients
                result["intent"] = "patient_specific_not_found"
                result["unresolved_refs"] = unresolved_refs
        else:
            # Not enough patients mentioned, treat as single patient
            result["intent"] = "patient_specific"
            result["patient_reference"] = patient_refs[0] if patient_refs else None

    # If patient-specific intent (single patient)
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
                # SLM detected patient intent but we can't find the patient
                result["intent"] = "patient_specific_not_found"
        else:
            # SLM says patient-specific but no name extracted
            # Check if we have a locked patient to use
            if locked_patient:
                result["resolved_patient"] = locked_patient
                result["intent"] = "patient_specific_use_locked"
            else:
                result["intent"] = "patient_specific_no_context"

    return result
