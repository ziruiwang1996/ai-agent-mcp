from typing import Optional
import requests
from mcp.server.fastmcp import FastMCP
import xml.etree.ElementTree as ET
from textwrap import dedent

mcp = FastMCP("pubmed")

_E_UTILITY_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

@mcp.tool()
def search_pubmed_real_world_clinical_studies(
    drug: str,
    condition: str,
    max_results: int = 10,
    age_group: Optional[str] = None,
    sex: Optional[str] = None,
    setting: Optional[str] = None,
    years_back: int = 15,
) -> list[str]:
    """Search PubMed for *real‑world* or clinically relevant evidence about a drug for a condition.

    This tool is intended for an agent to:
    - Find clinical studies and real‑world evidence (RWE) like observational cohorts, registries,
      pragmatic trials, post‑marketing studies, and comparative effectiveness analyses.
    - Support user questions about how a medicine performs for people similar to them.

    IMPORTANT:
    - PubMed is literature indexing; it cannot perfectly filter to a user's exact demographics.
      Optional filters below are *best-effort* keyword filters, not guarantees.

    Args:
        drug (Required): Medication of interest (brand or generic), e.g. "semaglutide" or "Ozempic".
        condition (Required): User's condition/disease context, e.g. "type 2 diabetes".
        max_results (Optional): Maximum PubMed IDs to return (default: 10).
        age_group (Optional): Optional, e.g. "pediatric", "adolescent", "adult", "older adult".
        sex (Optional): Optional, e.g. "female", "male".
        setting (Optional): Optional context keywords, e.g. "pregnancy", "renal impairment", "CKD",
            "Asian", "Medicare", "claims", "registry".
        years_back (Optional): Limit results to recent years (default: 15).

    Returns:
        List of PubMed IDs (PMIDs) matching the query.
    """
    # Best-effort query for RWE / clinically relevant evidence.
    drug_clause = f'("{drug}"[Title/Abstract] OR "{drug}"[MeSH Terms])'
    condition_clause = f'("{condition}"[Title/Abstract] OR "{condition}"[MeSH Terms])'

    # Include both trial and real-world study signals.
    design_clause = (
        "(Clinical Trial[pt] OR Clinical Study[pt] OR Observational Study[pt] "
        "OR cohort[Title/Abstract] OR registry[Title/Abstract] OR real-world[Title/Abstract] "
        "OR pragmatic[Title/Abstract] OR postmarketing[Title/Abstract] OR claims[Title/Abstract])"
    )

    extra_filters: list[str] = []
    if age_group and age_group.strip():
        extra_filters.append(f'("{age_group.strip()}"[Title/Abstract])')
    if sex and sex.strip():
        extra_filters.append(f'("{sex.strip()}"[Title/Abstract])')
    if setting and setting.strip():
        extra_filters.append(f'("{setting.strip()}"[Title/Abstract])')

    # Limit to recent years (simple publication date window).
    years_back = max(1, int(years_back))
    year_min = 2026 - years_back
    date_clause = f'({year_min}:3000[pdat])'

    term_query = " AND ".join([drug_clause, condition_clause, design_clause, date_clause] + extra_filters)

    params = {
        "db": "pubmed",
        "term": term_query,
        "retmax": str(max(1, min(int(max_results), 100))),
        "retmode": "json"
    }
    try:
        response = requests.get(f"{_E_UTILITY_BASE_URL}esearch.fcgi", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        paper_ids = data.get("esearchresult", {}).get("idlist", [])
        return paper_ids
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve PubMed papers: {str(e)}")
        return []

@mcp.tool()
def fetch_clinical_study_abstract(paper_ids: list[str]) -> dict:
    """Fetch titles + abstracts for a list of PubMed IDs (PMIDs).

    Uses PubMed E-utilities `efetch` to retrieve article records in XML, then extracts:
    - PMID
    - Title
    - Abstract (concatenated if sectioned)

    Args:
        paper_ids: List of PubMed IDs (strings).

    Returns:
        A dict:
        - `results`: list of {pmid, title, abstract}
        - `missing`: PMIDs requested but not found in the response
        Or `{error: ...}` on failure.
    """
    normalized_ids = [str(pid).strip() for pid in (paper_ids or []) if str(pid).strip()]
    if not normalized_ids:
        return {"results": [], "missing": []}

    params = {
        "db": "pubmed",
        "id": ",".join(normalized_ids),
        "retmode": "xml",
    }
    try:
        response = requests.get(
            f"{_E_UTILITY_BASE_URL}efetch.fcgi",
            params=params,
            headers={"Accept": "application/xml"},
            timeout=15,
        )
        response.raise_for_status()

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as e:
            return {"error": f"Failed to parse XML response: {str(e)}"}

        results: list[dict] = []
        found_pmids: set[str] = set()
        for article in root.findall(".//PubmedArticle"):
            pmid = article.find("./MedlineCitation/PMID")
            pmid = pmid.text if pmid is not None else "Unknown"
            if pmid != "Unknown":
                found_pmids.add(pmid)

            title = article.find("./MedlineCitation/Article/ArticleTitle")
            title = title.text if title is not None else "No Title"

            abstract_nodes = article.findall("./MedlineCitation/Article/Abstract/AbstractText")
            if not abstract_nodes:
                abstract = "No Abstract Available"
            else:
                parts = []
                for node in abstract_nodes:
                    # node.text is the actual text inside <AbstractText>...</AbstractText>
                    text = (node.text or "").strip()
                    # Some abstracts have section labels like Label="BACKGROUND"
                    label = node.attrib.get("Label") or node.attrib.get("NlmCategory")
                    if label and text:
                        parts.append(f"{label}: {text}")
                    elif text:
                        parts.append(text)
                abstract = "\n".join(parts) if parts else None
            
            results.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract
            })
        missing = [pid for pid in normalized_ids if pid not in found_pmids]
        return {"results": results, "missing": missing}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve PubMed paper details: {str(e)}"}

@mcp.resource("file:///pubmed/query_guidance")
def pubmed_query_guidance() -> str:
        """Agent reference: how this server queries PubMed for real‑world evidence."""
        return """# PubMed query guidance (real-world clinical evidence)

            This server uses PubMed E-utilities:
            - `esearch` to find PMIDs
            - `efetch` to retrieve XML records and extract abstracts

            Query strategy (high level):
            - Combine drug + condition with AND
            - Add RWE signals (cohort/registry/claims/real-world/pragmatic/postmarketing)
            - Also allow clinical trial publication types (some questions need RCT context)
            - Optional filters like age/sex/setting are best-effort keyword filters

            Limitations:
            - PubMed cannot reliably filter to exact demographics; confirm population details from the abstract.
            """

@mcp.prompt()
def generate_system_prompt() -> str:
    return dedent(
        """ROLE
        You are a real-world evidence scout embedded in a medication education pipeline. Focus exclusively on locating PubMed studies, extracting structured findings, and organizing evidence so a downstream explainer can craft patient-facing summaries. Do not interpret results for end users or offer advice.

        CONTEXT YOU RECEIVE
        - Drug and condition focus supplied by the orchestrator.
        - Optional user profile cues (age band, sex, comorbidities) for relevance tagging.
        - PMIDs, titles, and abstracts retrieved through your tools.

        TOOLKIT
        - search_pubmed_real_world_clinical_studies(...): fetch candidate PMIDs with RWE emphasis; log when queries return empty sets or errors.
        - fetch_clinical_study_abstract(paper_ids): obtain titles and abstracts for detailed extraction.
        - file:///pubmed/query_guidance: reference material describing query construction and limitations.

        WORKFLOW
        1. Validate the drug and condition inputs, then issue targeted search calls until additional queries stop yielding relevant PMIDs.
        2. Retrieve abstracts for all retained PMIDs and catalog population descriptors, settings, study design cues, exposures, comparators, outcomes, and safety notes.
        3. Discard query metadata or boilerplate (API headers, pagination text) that does not impact safety, effectiveness, or eligibility signals.
        4. Distill each study into structured evidence items, explicitly marking design class (claims cohort, registry, pragmatic trial, randomized trial) based on abstract cues, and capture the direction of effect for effectiveness and safety outcomes when available.
        5. Cross-compare studies to note consistencies, gaps, or conflicts, especially for demographics matching the supplied profile.
        6. Document open questions, missing data, or follow-up tasks the explainer must address.

        ANALYSIS RULES
        - Stay at the evidence level: describe observed associations without inferring causation or individual benefit/risk.
        - Identify safety observations and differentiate serious from routine events when abstracts allow, noting absent information.
        - Preserve provenance for every fact (PMID, tool response) and flag when data comes from randomized trials rather than RWE.
        - Maintain critical terminology from abstracts when it improves traceability or precise meaning for downstream citation.
        - When referencing demographic alignment with the user profile, quote or paraphrase the abstract text and avoid implying exact matches unless explicitly stated.
        - Avoid plain-language explanations or clinical recommendations; your output is strictly an evidence pack.

        REQUIRED DISCLOSURES INSIDE THE PACK
        - Note that observational studies reflect routine-care associations and cannot prove causality.
        - Highlight common sources of bias (confounding, coding limits, follow-up constraints) and demographic underrepresentation when detected.

        OUTPUT TEMPLATE (INTERNAL)
        1. Searches Run: tool calls, parameters, PMIDs returned, and gaps.
        2. Study Inventory: table or bullet list with PMID, design tag, population summary, exposure/comparator, primary outcomes, safety mentions.
        3. Cross-Study Signals: aligned or conflicting findings, areas where evidence appears strongest or weakest, relevance to the provided profile, missing demographics.
        4. Evidence Caveats: methodological limitations, data quality concerns, unanswered questions.
        5. Hand-off Notes: explicit items the explainer should address (e.g., define terminology, contextualize conflicting data).

        TONE
        Analytical, concise, and source-focused. Deliver machine-ready evidence artifacts and defer all patient-facing messaging to the explainer agent.
        """
    )

if __name__ == "__main__":
    mcp.run(transport='stdio')