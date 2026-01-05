import requests
import xml.etree.ElementTree as ET
from mcp.server.fastmcp import FastMCP
import re
import html as html_lib
from dotenv import load_dotenv
import os 

mcp = FastMCP("drug_label")

_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
_LABEL_SEARCHABLE_FIELDS_PATH = os.path.join(
    _ASSETS_DIR, "drug_labeling_fields.yaml"
)
_OPENFDA_LABEL_BASE_URL = "https://api.fda.gov/drug/label.json"
_MEDLINEPLUS_HEALTHTOPIC_BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"
_MERRIAM_WEBSTER_MEDICAL_DICTIONARY_BASE_URL = "https://www.dictionaryapi.com/api/v3/references/medical/json"

@mcp.resource("file:///openfda//label_searchable_fields")
def get_searchable_fields_of_drug_labeling():
    """Return openFDA Drug Labeling searchable field documentation (YAML).

    This is intended to be used by the agent as a reference for building valid
    `search` queries against https://api.fda.gov/drug/label.json.
    """
    try:
        with open(_LABEL_SEARCHABLE_FIELDS_PATH, "r", encoding="utf-8") as f:
            yaml_text = f.read()
        return "# openFDA drug product labeling searchable fields\n\n" + yaml_text
    except FileNotFoundError:
        return (
            "# openFDA drug product labeling searchable fields\n\n"
            "Searchable-fields YAML not found.\n\n"
            f"Expected at: {_LABEL_SEARCHABLE_FIELDS_PATH}"
        )

@mcp.tool()
def get_drug_information(name: str) -> dict:
    """
    Get FDA drug label information including warnings, side effects, dosage, and contraindications.
    Searches by brand name, generic name, or substance name.

    Args:
        name: Drug name to search (e.g., "Aspirin", "Ibuprofen", "Dupixent").
        
    Returns:
        Dict with FDA label data including warnings, adverse_reactions, dosage_and_administration,
        contraindications, indications_and_usage, and other label sections.
    """
    params = {
        "search": f'openfda.brand_name:"{name}" OR openfda.generic_name:"{name}" OR openfda.substance_name:"{name}"',
        "limit": 1
    }
    try:
        response = requests.get(_OPENFDA_LABEL_BASE_URL, params=params)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve drug information: {str(e)}"}
        
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
def generate_full_label_summarization_prompt(drug_name: str) -> str:
    """
    Generate a robust prompt for summarizing an FDA drug label.

    Args:
        drug_name: Name of the drug to interpret label for.
    """
    return f"""You are a medical information assistant. Your task is to summarize and interpret the FDA drug label for: {drug_name}

        Tooling (required):
        - First call: get_drug_information(name="{drug_name}")

        Data handling rules:
        - Use ONLY the first result from the tool response.
        - Do not invent label content. If a section is missing, say "Not found in returned label".
        - Prefer direct wording from the label when stating warnings/contraindications/dosing limits.

        Output format (use these headings exactly):

        1) Patient-friendly summary
        - What it is for (Indications and Usage)
        - How to take/use it (Dosage and Administration)
        - Who should NOT use it (Contraindications)
        - Biggest safety warnings (Boxed Warning, Warnings and Precautions)
        - Common side effects (Adverse Reactions)
        - Important interactions (Drug Interactions)
        - When to get urgent help (list red-flag symptoms; keep it conservative)

        2) Clinician summary (label-grounded)
        - Bullets grouped by section name (e.g., "INDICATIONS AND USAGE:")
        - Include key dosing parameters if present (route, frequency, maximums, renal/hepatic adjustments if returned)
        - Call out boxed warning(s), contraindications, and major interactions

        3) Safety highlights (one screen)
        - Contraindications
        - Boxed warning(s)
        - Top 3–6 serious risks / precautions
        - Top 3–6 high-impact interactions
        - Key dosing limits

        If the tool returns no results or an error:
        - Output exactly: No FDA label data found for {drug_name}.

        If you need to explain a medical term or condition context:
        - Use get_medical_term_definition(term="<term>") for brief definitions
        - Use get_health_topic_summary(term="<condition>") for general context
        - Do not make up explanations.

        Important: This is informational and not medical advice. Encourage users to follow the prescribing information and consult a clinician for decisions."""

@mcp.prompt()
def generate_label_section_interpretation_prompt(drug_name: str, section_name: str, section_content: str) -> str:
    """
    Generate a prompt for patient-friendly interpretation of one FDA label for a given section.

    Args:
        drug_name: Drug name for context.
        section_name: Name of the label section (e.g., "Indications_and_usage", "Dosage_and_administration").
        section_content: Raw text content of the label section.
    """
    return f"""You are a medical information assistant. Explain this FDA drug label section in plain language for a patient/caregiver.

        Drug: {drug_name}
        Section: {section_name}

        Section text:
        {section_content}

        Rules:
        - Use ONLY the section text above. Do not invent details (dose, frequency, risks, populations, outcomes).
        - Be clear, calm, and safety-first.
        - If the section text is incomplete/unclear, say what is missing and recommend asking a clinician/pharmacist.
        - If need additional drug information context, use get_drug_information(name="{drug_name}").

        Section-specific focus:
        - If this is an Indications/Usage-type section: explain what the medicine is for AND what it is not for (if stated).
        - If this is a Dosage/Administration-type section: turn the instructions into a simple checklist (route, timing, amount, maximums, missed dose rules if present). Do not guess.
        - If this is a Warnings/Contraindications-type section: explain who should avoid it and what warning signs require medical advice.

        Required output must satisfy ALL of the following:

        1) What this section means
        - 3–7 bullets in plain language explaining section content.
        - Please make the language clear and concise, ideally word count less than original text word count (use get_word_count(text="<text>") to check).

        2) Key terms explaination (when applicable)
        - Define up medical terms that appear in the section in parenthesis following their first mention.
        - If you need definitions, call: get_medical_term_definition(term="<term>").

        Recommended output (if applicable):

        1) What you should do
        - 3–8 actionable bullets (what to do / what to avoid / what to tell your care team).

        2) When to call a clinician or get urgent help
        - List red flags mentioned in the section. If none are mentioned, say: "This section does not list specific emergency warning signs."

        If the user asks about the condition itself (not the medicine):
        - You may call get_health_topic_summary(term="<condition>") for context; do not invent medical explanations.

        Important: This is informational and not medical advice."""

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')