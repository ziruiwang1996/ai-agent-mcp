from mcp.server.fastmcp import FastMCP
import requests
import os
from textwrap import dedent
from typing import Any, Optional

mcp = FastMCP("adverse_event")

_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
_FAERS_SEARCHABLE_FIELDS_PATH = os.path.join(
    _ASSETS_DIR, "drug_adverse_event_fields.yaml"
)
_FAERS_BASE_URL = "https://api.fda.gov/drug/event.json"
# Reuse TCP connections across calls for lower latency / CPU.
_HTTP_SESSION = requests.Session()
_DEFAULT_TIMEOUT_S = 20

@mcp.resource("file:///openfda/faers_searchable_fields")
def get_fields_of_FAERS_dataset() -> str:
    """Return FDA Adverse Event Reporting System (FAERS) searchable field documentation (YAML).
    This is intended to be used by the agent as a reference for building valid `search` queries 
    against https://api.fda.gov/drug/event.json, and for understanding the structure of the 
    returned FAERS records.
    """
    try:
        with open(_FAERS_SEARCHABLE_FIELDS_PATH, "r", encoding="utf-8") as f:
            yaml_text = f.read()
        return "# FDA Adverse Event Reporting System searchable fields\n\n" + yaml_text
    except FileNotFoundError:
        return (
            "# FDA Adverse Event Reporting System searchable fields\n\n"
            "Searchable-fields YAML not found.\n\n"
            f"Expected at: {_FAERS_SEARCHABLE_FIELDS_PATH}"
        )

def _openfda_get_json(params: dict[str, Any], *, timeout_s: int = _DEFAULT_TIMEOUT_S) -> dict:
    """Call openFDA FAERS endpoint and return a JSON object."""
    try:
        resp = _HTTP_SESSION.get(_FAERS_BASE_URL, params=params, timeout=timeout_s)
        if resp.status_code == 404:
            # openFDA uses 404 for no hits.
            return {"meta": {"results": {"total": 0}}, "results": []}
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve FAERS data: {str(e)}"}
    
def extract_report_from_raw_json(results: list[dict]) -> list[dict]:
    """Extract a compact, stable subset of fields from FAERS report records."""
    reports: list[dict] = []
    for report in results or []:
        patient = report.get("patient") or {}

        drugs_in = patient.get("drug") or []
        drugs = [
            {
                "actiondrug": drug.get("actiondrug", ""),
                "drugstartdate": drug.get("drugstartdate", ""),
                "drugstartdateformat": drug.get("drugstartdateformat", ""),
                "drugindication": drug.get("drugindication", ""),
                "medicinalproduct": drug.get("medicinalproduct", ""),
                "spl_set_id": (drug.get("openfda") or {}).get("spl_set_id", []),
            }
            for drug in drugs_in
            if isinstance(drug, dict)
        ]

        reports.append(
            {
                "safetyreportid": report.get("safetyreportid", ""),
                "serious": report.get("serious", ""),
                "seriousnesscongenitalanomali": report.get("seriousnesscongenitalanomali", ""),
                "seriousnessdeath": report.get("seriousnessdeath", ""),
                "seriousnessdisabling": report.get("seriousnessdisabling", ""),
                "seriousnesshospitalization": report.get("seriousnesshospitalization", ""),
                "seriousnesslifethreatening": report.get("seriousnesslifethreatening", ""),
                "seriousnessother": report.get("seriousnessother", ""),
                "transmissiondate": report.get("transmissiondate", ""),
                "transmissiondateformat": report.get("transmissiondateformat", ""),
                "patient": {
                    "patientonsetage": patient.get("patientonsetage", ""),
                    "patientonsetageunit": patient.get("patientonsetageunit", ""),
                    "patientsex": patient.get("patientsex", ""),
                    "patientweight": patient.get("patientweight", ""),
                    "patientagegroup": patient.get("patientagegroup", ""),
                    "reaction": patient.get("reaction", []),
                    "patientdeath": patient.get("patientdeath", {}),
                    "drug": drugs,
                },
            }
        )
    return reports

def _split_csv(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    items = [v.strip() for v in str(value).split(",")]
    items = [v for v in items if v]
    return items or None

def _lucene_quote(term: str) -> str:
    """Quote a term for openFDA/Lucene queries.
    openFDA supports quoting values with spaces. This also escapes any embedded
    double-quotes to avoid breaking the query.
    """
    term = term.lower().strip()
    return '"' + term.replace('\\', '\\\\').replace('"', '\\"') + '"'

def formulate_search_query(
    set_id: str,
    *,
    age: Optional[int] = None,
    age_window: Optional[int] = None,
    weight: Optional[float] = None,
    weight_window: Optional[float] = None,
    sex: Optional[str] = None,
    conditions: Optional[list[str]] = None,
    other_medications: Optional[list[str]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Build an openFDA Lucene `search` string for the FAERS endpoint.

    Returns:
        (search_query, error_message)
    """
    if not set_id:
        return None, "Missing required set_id."
    
    query_terms: list[str] = [f'patient.drug.openfda.spl_set_id:"{set_id}"']

    if age is not None:
        try:
            age_val = int(age)
        except (TypeError, ValueError):
            return None, "Invalid age; must be an integer."
        if age_val < 0:
            return None, "Invalid age; must be >= 0."
        window = 5 if age_window is None else int(age_window)
        if window < 0:
            return None, "Invalid age_window; must be >= 0."
        age_min = max(0, age_val - window)
        age_max = age_val + window
        query_terms.extend(
            [
                f"patient.patientonsetage:[{age_min} TO {age_max}]",
                "patient.patientonsetageunit:801",  # years
            ]
        )

    if weight is not None:
        try:
            weight_val = float(weight)
        except (TypeError, ValueError):
            return None, "Invalid weight; must be a number."
        if weight_val <= 0:
            return None, "Invalid weight; must be > 0."
        window = 5.0 if weight_window is None else float(weight_window)
        if window < 0:
            return None, "Invalid weight_window; must be >= 0."
        weight_min = max(0.0, weight_val - window)
        weight_max = weight_val + window
        query_terms.append(f"patient.patientweight:[{weight_min} TO {weight_max}]")

    if sex:
        sex_map = {
            "male": "1",
            "m": "1",
            "female": "2",
            "f": "2",
            "unknown": "0",
            "u": "0",
            "0": "0",
            "1": "1",
            "2": "2",
        }
        normalized_sex = sex_map.get(str(sex).strip().lower())
        if normalized_sex is None:
            return None, "Invalid sex; use male/female/unknown or 0/1/2."
        query_terms.append(f"patient.patientsex:{normalized_sex}")

    if conditions:
        terms = [_lucene_quote(c) for c in conditions if c]
        if terms:
            # openFDA grouping syntax: field:(a OR b OR c)
            query_terms.append("patient.drug.drugindication:(" + " OR ".join(terms) + ")")

    if other_medications:
        terms = [_lucene_quote(m) for m in other_medications if m]
        if terms:
            # openFDA grouping syntax: field:(a OR b OR c)
            query_terms.append("patient.drug.medicinalproduct:(" + " OR ".join(terms) + ")")

    return " AND ".join(query_terms), None

@mcp.tool()
def get_adverse_event_reports(
    set_id: Optional[str] = None,
    *,
    age: Optional[int] = None,
    age_window: Optional[int] = None,
    weight: Optional[float] = None,
    weight_window: Optional[float] = None,
    sex: Optional[str] = None,
    conditions: Optional[str] = None,
    other_medications: Optional[str] = None,
) -> dict:
    """Fetch FAERS adverse event reports filtered by patient/drug attributes.
    This calls the openFDA FAERS endpoint and returns a compact subset of fields
    (not the full raw report) for each matching case.

    Args:
        set_id (Required): SPL set ID to match against `patient.drug.openfda.spl_set_id`.
        age (Optional): Approximate onset age in years. Optional.
        age_window (Optional): +/- years around `age` for matching.
        weight (Optional): Approximate patient weight in kilograms.
        weight_window (Optional): +/- kg around `weight` for matching.
        sex (Optional): Patient sex as "male"/"female"/"unknown" or "1"/"2"/"0".
        conditions (Optional): Comma-separated list of medical conditions (indications) to filter on.
        other_medications (Optional): Comma-separated list of other medications to filter on.

    Notes:
        - FAERS is a spontaneous reporting system with reporting bias and missing data.
        - Results do not establish causality.

    Returns:
        A dict with `search_query`, `total_reports`, and `reports` on success, or `error`.
    """
    if not set_id:
        return {"error": "Missing required set_id parameter."}

    conditions_list = _split_csv(conditions)
    other_meds_list = _split_csv(other_medications)

    search_query, err = formulate_search_query(
        set_id=set_id,
        age=age,
        age_window=age_window,
        weight=weight,
        weight_window=weight_window,
        sex=sex,
        conditions=conditions_list,
        other_medications=other_meds_list,
    )
    if err:
        return {"error": err}

    params = {
        "search": search_query,
        "limit": 10,
    }
    data = _openfda_get_json(params)
    if "error" in data:
        return data

    results = data.get("results") or []
    reports = extract_report_from_raw_json(results)
    return {
        "set_id": set_id,
        "filters": {
            "age": age,
            "age_window": age_window,
            "weight": weight,
            "weight_window": weight_window,
            "sex": sex,
            "conditions": conditions_list,
            "other_medications": other_meds_list,
        },
        "search_query": search_query,
        "total_reports": data.get("meta", {}).get("results", {}).get("total"),
        "reports": reports,
    }

@mcp.tool()
def get_adverse_reaction_count(
    set_id: Optional[str] = None,
    *,
    age: Optional[int] = None,
    age_window: Optional[int] = None,
    weight: Optional[float] = None,
    weight_window: Optional[float] = None,
    sex: Optional[str] = None,
    conditions: Optional[str] = None,
    other_medications: Optional[str] = None,
) -> dict:
    """Return a frequency table of reported reactions (MedDRA PT) for matching FAERS cases.
    This uses openFDA's `count=` aggregation, so it is more memory-efficient than
    fetching and parsing full case reports.

    Args:
        set_id (Required): SPL set ID to match against `patient.drug.openfda.spl_set_id`.
        age (Optional): Approximate onset age in years.
        age_window (Optional): +/- years around `age` for matching.
        weight (Optional): Approximate patient weight in kilograms.
        weight_window (Optional): +/- kg around `weight` for matching.
        sex (Optional): Patient sex as "male"/"female"/"unknown" or "1"/"2"/"0".
        conditions (Optional): Comma-separated list of medical conditions (indications) to filter on.
        other_medications (Optional): Comma-separated list of other medications to filter on.

    Returns:
        A dict with `search_query`, `total_reports`, and `results` (reaction counts) on success, or `error`.  
    """
    if not set_id:
        return {"error": "Missing required set_id parameter."}

    conditions_list = _split_csv(conditions)
    other_meds_list = _split_csv(other_medications)

    search_query, err = formulate_search_query(
        set_id=set_id,
        age=age,
        age_window=age_window,
        weight=weight,
        weight_window=weight_window,
        sex=sex,
        conditions=conditions_list,
        other_medications=other_meds_list,
    )
    if err:
        return {"error": err}
    
    params = {
        "search": search_query,
        "count": "patient.reaction.reactionmeddrapt.exact"
    }

    data = _openfda_get_json(params, timeout_s=15)
    if "error" in data:
        return data

    return {
        "set_id": set_id,
        "filters": {
            "age": age,
            "age_window": age_window,
            "weight": weight,
            "weight_window": weight_window,
            "sex": sex,
            "conditions": conditions_list,
            "other_medications": other_meds_list,
        },
        "search_query": search_query,
        "total_reports": data.get("meta", {}).get("results", {}).get("total"),
        "results": data.get("results") or [],
    }

@mcp.prompt()
def generate_system_prompt() -> str:
    return dedent(
        """ROLE
        You are an adverse-event evidence collector embedded in a patient-facing medication education workflow. Focus exclusively on locating, analyzing, and distilling FAERS evidence relevant to the provided drug context and user profile. Do not craft patient-facing explanations or recommendations—downstream agents own messaging.

        CONTEXT YOU RECEIVE
        • Drug identifiers (name and set_id).
        • Patient demographics and concomitant medications when available.
        • FAERS payloads and reference material returned via your tools.

        TOOLKIT
        • get_adverse_event_reports(...): retrieves structured case summaries plus hit counts.
        • get_adverse_reaction_count(...): retrieves MedDRA Preferred Term frequency tables.
        • file:///openfda/faers_searchable_fields: YAML reference for valid openFDA fields and syntax.

        WORKFLOW
        1. Validate incoming drug context and patient filters, then issue targeted tool calls to capture cohorts matching those attributes.
        2. Audit returned data for consistency, documenting empty, partial, or conflicting responses.
        3. Extract and tag evidence: reaction clusters, seriousness signals, demographic notes, and any comedication patterns.
        4. Ignore API boilerplate (metadata, pagination notices) and focus on clinically meaningful content.
        5. Package findings into a structured evidence bundle that preserves provenance (tool name, query parameters, FAERS references) for downstream explanation.

        ANALYSIS RULES
        • Keep analysis at the cohort-level; never infer causation, risk, or advice for individuals.
        • Clarify how similarity to the user was determined (age range, sex, concomitant therapies) and mark approximations or unknowns.
        • Separate serious outcomes from routine events and flag coexisting conditions or medications when evident.
        • Cross-reference supplied drug-label insights only to note alignment or divergence, without asserting new safety conclusions.
        • Maintain professional boundaries: no medical guidance, dosing opinions, or probability estimates.
        • Preserve critical terminology and phrasing from FAERS where it enhances traceability for downstream citation.

        REQUIRED DISCLOSURES WITHIN THE BUNDLE
        • Record that FAERS is a voluntary reporting system and does not prove causation.
        • Note that FAERS data can be incomplete, duplicated, or biased, and cannot establish incidence or individual predictions.

        OUTPUT TEMPLATE (EVIDENCE-ONLY)
        1. Sources Queried: list tool calls, parameters, and response status (hits, empty, errors).
        2. Cohort Snapshot: summarize matched cohorts, total reports, and relevance to the user profile.
        3. Evidence Items:
           - Reaction Patterns: grouped MedDRA PTs with seriousness tiers and qualitative prevalence cues from counts.
           - Case Highlights: notable report attributes (e.g., concomitant meds, age clusters, repeat signals) with FAERS references.
           - Label Cross-Checks: observed overlaps or discrepancies versus provided label data.
        4. Data Caveats & Follow-ups: document gaps, potential duplicates, or additional queries needed for the explainer agent.

        TONE
        Operational, factual, and concise. Deliver machine-ready evidence artifacts without interpretive narrative.
        """
    )

if __name__ == "__main__":
    mcp.run(transport='stdio')