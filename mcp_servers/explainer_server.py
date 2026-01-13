from mcp.server.fastmcp import FastMCP
import requests
import xml.etree.ElementTree as ET
import html as html_lib
import re
from dotenv import load_dotenv
import os
from textwrap import dedent

mcp = FastMCP("explainer")

_MEDLINEPLUS_HEALTHTOPIC_BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"
_MERRIAM_WEBSTER_MEDICAL_DICTIONARY_BASE_URL = "https://www.dictionaryapi.com/api/v3/references/medical/json"

@mcp.tool()
def get_health_topic_summary(term: str, clean_html: bool = True) -> str:
    """
    Get comprehensive summary about a medical condition, disease, or health topic from NLM MedlinePlus.
    Use this for general health information, condition overviews, and patient education content.
    
    Args:
        term: Medical condition or health topic (e.g., "diabetes", "asthma", "hypertension").
        clean_html: Whether to remove HTML tags from output (default: True).
    
    Returns:
        Plain text summary of the health topic from authoritative medical sources.

    Agent guidance:
        If you need an explanation of medical information, call this tool.
        Do not make up explanations; base your summary on the returned content.
    """
    params = {
        "db": "healthTopics",
        "term": term,
        "rettype": "brief",
        "retmax": 1
    }
    try:
        response = requests.get(
            _MEDLINEPLUS_HEALTHTOPIC_BASE_URL, 
            params=params, 
            headers={"Accept": "application/xml"}, 
            timeout=15
        )
        response.raise_for_status()
        # Parse XML and traverse documents
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            return "Failed to parse XML response from NLM service."

        documents = root.findall('.//document')
        if not documents:
            return "No results found for the given term."

        summaries = []
        for doc in documents:
            # Extract only FullSummary content
            for content in doc.findall('./content'):
                name = content.attrib.get('name', '').strip()
                if name == 'FullSummary':
                    raw_text = (content.text or '').strip()
                    if clean_html and raw_text:
                        # Remove HTML tags and unescape entities
                        text = re.sub(r"<[^>]+>", "", raw_text)
                        text = html_lib.unescape(text).strip()
                    else:
                        text = raw_text
                    if text:
                        summaries.append(text)
                    break

        if summaries:
            return "\n\n".join(summaries)
        else:
            return "No FullSummary content found in XML response."
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve medical term definition: {str(e)}"}
        
@mcp.tool()
def get_medical_term_definition(term: str) -> str:
    """
    Get concise dictionary definition of a medical term from Merriam-Webster Medical Dictionary.
    Use this to define medical terminology, anatomical terms, or clinical vocabulary.
    
    Args:
        term: Medical term to define (e.g., "hypertension", "myocardial", "thrombosis").
    
    Returns:
        Concise definition(s) of the medical term, one per line.

    Agent guidance:
        Use this when you need to define a medical term.
        Do not invent definitions; return only content derived from the source.
    """
    url = f"{_MERRIAM_WEBSTER_MEDICAL_DICTIONARY_BASE_URL}/{term}?"

    load_dotenv()
    if not os.getenv("MERRIAM_WEBSTER_API_KEY"):
        return "MERRIAM_WEBSTER_API_KEY not found in environment variables."
    params = {
        "key": os.getenv("MERRIAM_WEBSTER_API_KEY")
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            return "No definition found for the given term."

        defs = []
        for result in data:
            if isinstance(result, dict) and "shortdef" in result:
                shortdef = result["shortdef"]
                defs.extend(shortdef if isinstance(shortdef, list) else [str(shortdef)])
        return "\n".join(defs) if defs else "No definition found for the given term."
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve medical term definition: {str(e)}"}

@mcp.prompt()
def generate_system_prompt() -> str:
    return dedent(
        """ROLE
        You are the explainer agent in a patient-facing medication education experience. Your job is to transform the upstream evidence bundles (FAERS, clinical trials, RWE, drug labels, reference material) into a coherent, plain-language narrative for the user. Do not introduce new facts—rely only on the supplied evidence packs and authorized reference tools.

        INPUTS
        - Structured evidence bundles from collector agents (adverse events, clinical trials, drug labels, PubMed/RWE, relevance filter).
        - Optional user context: demographics, medications, conditions, questions.
        - Reference tools: get_health_topic_summary and get_medical_term_definition for supplemental explanations only when needed.

        WORKFLOW
        1. Review all evidence bundles; note provenance tags (FAERS, ClinicalTrials.gov, PubMed, FDA label) so you can attribute insights.
        2. Identify the core question or concern and map which evidence elements answer it (e.g., safety signals, effectiveness themes, eligibility guidance).
        3. Translate technical content into accessible language, grouping reactions or findings into intuitive themes or seriousness tiers.
        4. Integrate overlapping evidence (e.g., FAERS vs label warnings vs real-world studies) and highlight consistencies or tensions without speculating.
        5. Call get_medical_term_definition for unfamiliar terminology and get_health_topic_summary when high-level condition context is required. Cite that the information comes from these sources.
        6. Flag any missing or uncertain data surfaced by upstream agents and recommend professional follow-up when appropriate.

        RESPONSE STRUCTURE
        1. Snapshot: succinct overview of the evidence reviewed (sources, cohorts, recency).
        2. What Was Observed: patient-friendly summary of effectiveness or usage insights, grouped by theme.
        3. Safety Watchpoints: serious vs routine reactions, including source attribution (FAERS, trials, label) and reassurance about voluntary reporting limitations.
        4. How This Fits the User: explain approximate relevance to the user profile while stressing variability.
        5. Limitations & Next Steps: call out data gaps, study constraints, and encourage consultation with a healthcare professional for personalized guidance.

        COMMUNICATION RULES
        - Maintain calm, empathetic, non-diagnostic tone; use second person sparingly and respectfully.
        - Never provide medical advice, treatment decisions, or dosing instructions. Encourage consultation with clinicians for personal questions.
        - Explicitly state limitations of each data source (FAERS voluntary reporting, observational confounding, trial eligibility).
        - Preserve key numeric values, time frames, and named entities; quote terminology when exact phrasing matters, then clarify in parentheses if needed.
        - If evidence is missing or conflicting, say so and explain what that means for certainty.

        SAFETY & SOURCING
        - Attribute insights to their sources (e.g., “According to FAERS reports…”, “In ClinicalTrials.gov study NCT…”, “The FDA label states…”).
        - If tools return no relevant content, state that clearly and suggest verifying with a healthcare professional.
        - Use get_medical_term_definition and get_health_topic_summary only when clarification is required; do not copy large passages verbatim.

        CLOSING
        - Finish by reminding users that this information supplements, not replaces, discussions with their healthcare team.
        """
    )


if __name__ == "__main__":
    mcp.run(transport='stdio')