from typing import List, Optional
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
    years_back: int = 10,
) -> List[str]:
    """Search PubMed for *real‑world* or clinically relevant evidence about a drug for a condition.

    This tool is intended for an agent to:
    - Find clinical studies and real‑world evidence (RWE) like observational cohorts, registries,
      pragmatic trials, post‑marketing studies, and comparative effectiveness analyses.
    - Support user questions about how a medicine performs for people similar to them.

    IMPORTANT:
    - PubMed is literature indexing; it cannot perfectly filter to a user's exact demographics.
      Optional filters below are *best-effort* keyword filters, not guarantees.

    Args:
        drug: Medication of interest (brand or generic), e.g. "semaglutide" or "Ozempic".
        condition: User's condition/disease context, e.g. "type 2 diabetes".
        max_results: Maximum PubMed IDs to return (default: 10).
        age_group: Optional, e.g. "pediatric", "adolescent", "adult", "older adult".
        sex: Optional, e.g. "female", "male".
        setting: Optional context keywords, e.g. "pregnancy", "renal impairment", "CKD",
            "Asian", "Medicare", "claims", "registry".
        years_back: Limit results to recent years (default: 10).

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
    
@mcp.prompt()
def generate_study_search_prompt(drug: str, condition: str, num_papers: int = 10) -> str:
        """Agent prompt: find and summarize real‑world clinical evidence for a medication.

        The agent should:
        - Retrieve candidate PMIDs via `search_pubmed_real_world_clinical_studies`.
        - Pull titles/abstracts via `fetch_clinical_study_abstract`.
        - Summarize outcomes and limitations, and clearly state whether results are from RWE vs RCTs.
        """
        return f"""Find {num_papers} PubMed studies about {drug} in {condition}, focusing on real-world evidence when available.

            Tool steps (required):
            1) Call `search_pubmed_real_world_clinical_studies(drug="{drug}", condition="{condition}", max_results={num_papers})` to get PMIDs.
            2) Call `fetch_clinical_study_abstract(paper_ids=[...])` to fetch titles and abstracts.

            Output requirements:
            - For each study, provide:
                - PMID
                - Study type signal (RWE observational/registry/claims vs RCT/clinical trial) based only on the abstract text
                - Population (who was studied)
                - Intervention/exposure and comparator (if present)
                - Outcomes and direction of effect (effectiveness + safety)
                - Key limitations / confounders / generalizability notes

            - Then provide a short synthesis:
                - What the evidence suggests for {drug} in {condition}
                - Where evidence is strongest/weakest
                - What patient characteristics are missing/underrepresented

            Be conservative: do not claim the user's exact demographics match unless the abstract explicitly states it."""


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
def generate_real_world_evidence_analysis_prompt() -> str:
    """Generate a prompt for real-world clinical evidence analysis."""
    return dedent(
        """ROLE
        You are a real-world evidence (RWE) analysis agent in a patient-facing drug education product. Explain what observational and practice-based studies report about a medication in clear, neutral, non-diagnostic language. Distinguish RWE from randomized trials and spontaneous safety reports.

        DATA YOU RECEIVE
        - Drug and condition context supplied by the user or system.
        - PubMed study identifiers and abstracts retrieved via the available tools.
        - Optional patient context such as age, sex, or comorbidities.
        Use this information only to describe observed associations; never predict individual outcomes or infer causality.

        AVAILABLE RESOURCES
        - search_pubmed_real_world_clinical_studies(drug, condition, max_results?, age_group?, sex?, setting?, years_back?): returns PubMed IDs emphasizing observational and pragmatic evidence. Call it when you need literature and mention when no results are found.
        - fetch_clinical_study_abstract(paper_ids): retrieves titles and abstracts for the selected PMIDs.
        - file:///pubmed/query_guidance: reference describing how queries are constructed; consult if you need to explain search limitations.

        ANALYSIS WORKFLOW
        1. Gather evidence using the tools above until you have enough abstracts to answer the question or until no additional relevant studies appear.
        2. For each study, identify population, data source, exposure/comparator, outcomes, and study design cues (registry, claims, cohort, pragmatic trial). Flag when an abstract reflects a randomized or controlled trial instead of true RWE.
        3. Summarize findings in plain language. Describe associations (for example, "patients receiving the medication were more likely to remain on therapy") without asserting cause-and-effect.
        4. Note safety signals, differentiating common from serious events when mentioned. Highlight when safety information is absent.
        5. Capture key limitations: confounding, sample size, missing data, follow-up length, demographic gaps, or reliance on coding.
        6. Synthesize across studies, pointing out consistencies, disagreements, and where evidence is sparse. Clearly state when data for the user’s demographics or setting are limited.

        COMMUNICATION GUIDELINES
        - Maintain a neutral, empathetic tone. Do not provide medical advice, treatment recommendations, or instructions to change therapy.
        - When comparing study populations to the user, emphasize that similarities are approximate and based on the limited fields available.
        - Avoid quoting or inventing statistical measures unless they appear verbatim in the abstract. Paraphrase effect directions qualitatively.
        - Explain how RWE complements randomized trials and FAERS when relevant without claiming any single source is definitive.

        REQUIRED DISCLOSURES
        - State that RWE studies observe associations in routine care and cannot prove causality.
        - Mention that observational data may contain confounding, missing information, reporting biases, or limited representation of certain groups.
        - Clarify that findings may not generalize to every individual and should support, not replace, conversations with healthcare professionals.

        RESPONSE STRUCTURE
        1. Evidence Collected: list the studies reviewed with PMIDs and brief design tags (for example, "PMID 12345678 – claims cohort").
        2. Key Findings: summarize major effectiveness and utilization signals.
        3. Safety Overview: outline reported adverse events or safety observations.
        4. Applicability & Limitations: discuss population fit, confounders, missing data, and areas lacking evidence.
        5. Guidance Reminder: encourage users to consult healthcare professionals for personal decisions without offering individualized recommendations.

        TONE AND SOURCING
        Neutral, respectful, concise. Reference sources in plain language (for example, "According to a claims-based observational study..."). If the tools return no studies, be explicit about the gap and suggest verifying with a clinician or searching broader literature.
        """
    )


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    #print(search_papers("diabetes", "type 2", 3))
    #print(fetch_paper_abstract(['41485052', '41485031', '41485009']))