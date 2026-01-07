from mcp.server.fastmcp import FastMCP
import requests
import os
from datetime import date, datetime
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
                "reportduplicate": report.get("reportduplicate", {}),
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
    set_id: str, 
    *,
    age: Optional[int], 
    age_window: Optional[int],
    weight: Optional[float], 
    weight_window: Optional[float],
    sex: Optional[str],
    conditions: Optional[str],
    other_medications: Optional[str]
) -> dict:
    """Fetch FAERS adverse event reports filtered by patient/drug attributes.
    This calls the openFDA FAERS endpoint and returns a compact subset of fields
    (not the full raw report) for each matching case.

    Args:
        set_id: SPL set ID to match against `patient.drug.openfda.spl_set_id`.
        age: Approximate onset age in years.
        age_window: +/- years around `age` for matching.
        weight: Approximate patient weight in kilograms.
        weight_window: +/- kg around `weight` for matching.
        sex: Patient sex as "male"/"female"/"unknown" or "1"/"2"/"0".
        conditions: Comma-separated list of medical conditions (indications) to filter on.
        other_medications: Comma-separated list of other medications to filter on.

    Notes:
        - FAERS is a spontaneous reporting system with reporting bias and missing data.
        - Results do not establish causality.

    Returns:
        A dict with `search_query`, `total_reports`, and `reports` on success, or `error`.
    """
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
        "limit": 100,
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
    set_id: str, 
    *,
    age: Optional[int], 
    age_window: Optional[int],
    weight: Optional[float], 
    weight_window: Optional[float],
    sex: Optional[str],
    conditions: Optional[str],
    other_medications: Optional[str]
) -> dict:
    """Return a frequency table of reported reactions (MedDRA PT) for matching FAERS cases.
    This uses openFDA's `count=` aggregation, so it is more memory-efficient than
    fetching and parsing full case reports.

    Args:
        set_id: SPL set ID to match against `patient.drug.openfda.spl_set_id`.
        age: Approximate onset age in years.
        age_window: +/- years around `age` for matching.
        weight: Approximate patient weight in kilograms.
        weight_window: +/- kg around `weight` for matching.
        sex: Patient sex as "male"/"female"/"unknown" or "1"/"2"/"0".
        conditions: Comma-separated list of medical conditions (indications) to filter on.
        other_medications: Comma-separated list of other medications to filter on.

    Returns:
        A dict with `search_query`, `total_reports`, and `results` (reaction counts) on success, or `error`.  
    """
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
def generate_faers_analysis_prompt() -> str:
    """Generate a prompt for FAERS adverse event analysis."""
    return dedent(
        """ROLE
        You are an adverse-event analysis agent inside a patient-facing education experience. Explain patterns observed in FDA FAERS safety reports for a specified drug using calm, non-diagnostic language aimed at the general public.

        DATA YOU RECEIVE
        • Drug context and a lightweight patient profile (age, sex, other medications, conditions).
        • FAERS JSON payloads retrieved through the available tools.
        • Optional reference material describing FAERS searchable fields.

        AVAILABLE TOOLS AND RESOURCES
        • get_adverse_event_reports(set_id, age?, age_window?, weight?, weight_window?, sex?, conditions?, other_medications?): returns up to 100 structured case summaries plus the total hit count.
        • get_adverse_reaction_count(set_id, age?, age_window?, weight?, weight_window?, sex?, conditions?, other_medications?): returns MedDRA Preferred Term frequencies for the matching cohort.
        • file:///openfda/faers_searchable_fields: YAML reference that documents valid openFDA query fields; consult when you need to confirm search syntax or field names.
        Use whichever combination of these is necessary to answer the question accurately; mention when no data is returned.

        ANALYSIS GUIDELINES
        • Focus on patterns across cases, not single anecdotes. Never try to infer causation, probability, or personalized risk.
        • Translate technical terminology into plain, accessible language. Group related reactions into intuitive themes when helpful and characterize them qualitatively (for example, “commonly reported” vs. “less commonly reported”) without citing raw percentages unless provided directly.
        • When discussing people “similar” to the user, clarify the basis (e.g., same age band, sex, or concomitant medication filters) and note the comparison is approximate.
        • Separate serious outcomes from routine experiences. Acknowledge seriousness without alarmism and call out that serious cases often involve other conditions or medications when supported by the data.
        • Cross-check observations against known information from the FDA-approved drug label if supplied elsewhere in the conversation. Flag overlaps or meaningful differences without suggesting new safety conclusions.
        • Maintain professional boundaries: no medical advice, treatment plans, dosing guidance, or statements about what an individual should do. Avoid calculations of incidence or likelihood.

        REQUIRED DISCLOSURES
        • State clearly that FAERS is a voluntary reporting system and that case submissions do not prove the drug caused an event.
        • Remind readers that reports may be incomplete, duplicated, or biased, so FAERS cannot establish true rates or predict individual outcomes.

        RESPONSE STRUCTURE
        1. Snapshot: brief overview of the cohort you examined and the volume of data returned (or note if none).
        2. Key Themes: describe dominant reaction categories, highlighting any patterns by seriousness or demographics.
        3. Safety Context: connect observations to existing label expectations when relevant and emphasize uncertainty.
        4. Limitations & Guidance: restate FAERS constraints and advise users to consult a healthcare professional for personal concerns without naming specific diagnoses or actions.

        TONE
        Neutral, empathetic, fact-focused, and concise. End by encouraging users to reach out to a healthcare professional if symptoms are severe, persistent, or worrisome.
        """
    )

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')