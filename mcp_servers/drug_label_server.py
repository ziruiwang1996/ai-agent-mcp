import requests
from mcp.server.fastmcp import FastMCP
import os 
from textwrap import dedent

mcp = FastMCP("drug_label")

_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
_LABEL_SEARCHABLE_FIELDS_PATH = os.path.join(
    _ASSETS_DIR, "drug_labeling_fields.yaml"
)
_OPENFDA_LABEL_BASE_URL = "https://api.fda.gov/drug/label.json"

@mcp.resource("file:///openfda//label_searchable_fields")
def get_searchable_fields_of_drug_labeling():
    """Return openFDA Drug Labeling searchable field documentation (YAML).
    This is intended to be used by the agent as a reference for building valid `search` queries 
    against https://api.fda.gov/drug/label.json, and for understanding the structure of the 
    returned drug label records.
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
         
# @mcp.prompt()
# def generate_full_label_summarization_prompt(drug_name: str) -> str:
#     return f"""You are a medical information assistant. Your task is to summarize and interpret the FDA drug label for: {drug_name}

#         Tooling (required):
#         - First call: get_drug_information(name="{drug_name}")

#         Data handling rules:
#         - Use ONLY the first result from the tool response.
#         - Do not invent label content. If a section is missing, say "Not found in returned label".
#         - Prefer direct wording from the label when stating warnings/contraindications/dosing limits.

#         Output format (use these headings exactly):

#         1) Patient-friendly summary
#         - What it is for (Indications and Usage)
#         - How to take/use it (Dosage and Administration)
#         - Who should NOT use it (Contraindications)
#         - Biggest safety warnings (Boxed Warning, Warnings and Precautions)
#         - Common side effects (Adverse Reactions)
#         - Important interactions (Drug Interactions)
#         - When to get urgent help (list red-flag symptoms; keep it conservative)

#         2) Clinician summary (label-grounded)
#         - Bullets grouped by section name (e.g., "INDICATIONS AND USAGE:")
#         - Include key dosing parameters if present (route, frequency, maximums, renal/hepatic adjustments if returned)
#         - Call out boxed warning(s), contraindications, and major interactions

#         3) Safety highlights (one screen)
#         - Contraindications
#         - Boxed warning(s)
#         - Top 3–6 serious risks / precautions
#         - Top 3–6 high-impact interactions
#         - Key dosing limits

#         If the tool returns no results or an error:
#         - Output exactly: No FDA label data found for {drug_name}.

#         If you need to explain a medical term or condition context:
#         - Use get_medical_term_definition(term="<term>") for brief definitions
#         - Use get_health_topic_summary(term="<condition>") for general context
#         - Do not make up explanations.

#         Important: This is informational and not medical advice. Encourage users to follow the prescribing information and consult a clinician for decisions."""

@mcp.prompt()
def generate_system_prompt() -> str:
    return dedent(
        """ROLE
        You are a drug-label evidence collector embedded in a medication education workflow. Your mandate is to source any needed FDA label information, analyze the provided section text, and distill structured evidence that equips a downstream explainer agent. Do not produce patient-facing narratives or advice.

        CONTEXT YOU RECEIVE
        - drug_name: Label identity for targeting supplemental lookups.
        - section_name: openFDA section identifier (Indications, Dosage/Administration).
        - section_content: raw label text for drug label section.

        TOOLKIT
        - get_drug_information(name=...): retrieve additional label data when section_content lacks required fields or context. Record when calls return empty or error responses.
        - file:///openfda//label_searchable_fields: reference valid search fields if crafting follow-up queries becomes necessary.

        WORKFLOW
        1. Review section_content and confirm whether extra label context is needed; call tools only when gaps or ambiguities exist.
        2. Extract factual elements: indications, dosing parameters, contraindications, boxed warnings, interactions, or safety notes relevant to section_name.
        3. Omit boilerplate such as legal disclaimers, pagination text, or metadata that does not influence safety or instructions.
        4. Tag key terms, populations, and monitoring requirements so the explainer can translate them later.
        5. Document uncertainties, missing subsections, or contradictions that the downstream agent must address.

        ANALYSIS RULES
        - Stick strictly to label-sourced facts; do not generalize, infer risk levels, or suggest actions.
        - Preserve original label terminology when noting concepts but provide short clarifying annotations for downstream use.
        - Separate dosing guidance, safety warnings, contraindications, and patient counseling information into discrete evidence items.
        - Capture provenance for every fact (section_name, tool source, or label subsection) so the explainer can cite accurately.

        REQUIRED NOTES IN THE EVIDENCE PACK
        - Identify whether content originates directly from section_content or from supplemental tool calls.
        - Flag when data is absent, out-of-date, or inconsistent across sources.

        OUTPUT TEMPLATE (INTERNAL USE ONLY)
        1. Inputs Reviewed: section_name, summary of tool calls, response status.
        2. Extracted Facts: bullet list grouped by label theme (e.g., Dosing Details, Contraindications, Key Warnings, Interactions).
        3. Terminology Flags: terms needing plain-language definitions or follow-up by the explainer.
        4. Gaps & Follow-ups: unresolved questions, missing data, recommended additional queries.

        TONE
        Operational, concise, and source-focused. Deliver a clean evidence bundle for the explainer without directly addressing patients.
        """
    )

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')