from typing import Optional, Tuple, List, Dict, Any
import re
from patients import fuzzy_resolve

def extract_patient_references(query: str, roster: List[Dict[str, str]]) -> Tuple[Optional[str], List[str]]:
    """
    Extract potential patient names or IDs from a natural language query.
    Returns: (detected_name_or_id, potential_name_parts)
    """
    query_lower = query.lower()

    # Check for explicit patient ID patterns (e.g., "patient IVF001", "ID: IVF001")
    id_patterns = [
        r"patient\s+([A-Z0-9]+)",
        r"patient\s+id\s+([A-Z0-9]+)",
        r"id:?\s*([A-Z0-9]+)",
    ]
    for pattern in id_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1), []

    # Look for possessive patterns that might indicate a patient name
    # e.g., "Alex's", "show me Sarah's", "what are John's"
    possessive_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\b"
    possessive_matches = re.findall(possessive_pattern, query)
    if possessive_matches:
        return possessive_matches[0], possessive_matches

    # Look for "patient NAME" or "for NAME" patterns
    name_patterns = [
        r"patient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1), [match.group(1)]

    # Extract all capitalized words (potential names) but only if query suggests patient context
    patient_indicators = ["show me", "patient", "his", "her", "their"]
    has_patient_context = any(indicator in query_lower for indicator in patient_indicators)

    if has_patient_context:
        # Find capitalized words that might be names
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        # Filter out common medical terms
        common_terms = {"MRI", "CT", "Labs", "Results", "Report", "Dr", "Doctor"}
        potential_names = [w for w in capitalized_words if w not in common_terms]
        if potential_names:
            return " ".join(potential_names[:2]), potential_names  # Take up to 2 words as potential full name

    return None, []

def is_patient_specific_query(query: str) -> bool:
    """
    Determine if a query is likely asking about a specific patient.
    """
    patient_indicators = [
        "his", "her", "their", "this patient", "the patient",
        "show me", "what are", "patient's", "patient",
        "latest", "recent", "current", "last"
    ]

    query_lower = query.lower()
    return any(indicator in query_lower for indicator in patient_indicators)

def analyze_query(query: str, roster: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyze a query to determine intent and extract patient information.
    Returns a dict with:
        - intent: "patient_specific" or "general"
        - patient_reference: extracted name/ID if found
        - resolved_patient: patient dict if uniquely resolved
        - candidates: list of candidate patients if ambiguous
    """
    result = {
        "intent": "general",
        "patient_reference": None,
        "resolved_patient": None,
        "candidates": []
    }

    # Extract potential patient references
    patient_ref, _ = extract_patient_references(query, roster)

    if patient_ref:
        result["patient_reference"] = patient_ref
        # Try to resolve the patient
        resolved, candidates, reason = fuzzy_resolve(roster, patient_ref)
        if resolved:
            result["intent"] = "patient_specific"
            result["resolved_patient"] = resolved
        elif candidates:
            result["intent"] = "patient_specific"
            result["candidates"] = candidates
    elif is_patient_specific_query(query):
        # Query seems patient-specific but no name found
        result["intent"] = "patient_specific_no_context"

    return result