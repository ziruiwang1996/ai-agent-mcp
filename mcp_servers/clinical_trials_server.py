import requests
from mcp.server.fastmcp import FastMCP
from textwrap import dedent
from typing import Any, Optional

mcp = FastMCP("clinical_trials")

_CLINICAL_TRIALS_DOT_GOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
_HTTP_SESSION = requests.Session()

def _deep_get(obj: Any, path: str, default=None):
    """Safely read nested dict values using a dotted path."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return default
    return default if cur is None else cur

def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def _parse_age_years(age_text: Optional[str]) -> Optional[int]:
    """Parse strings like '18 Years' into an int (years) when possible."""
    if not age_text:
        return None
    s = str(age_text).strip()
    parts = s.split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None

def _sex_matches(trial_sex: Optional[str], user_sex: Optional[str]) -> Optional[bool]:
    """Return whether a trial's sex eligibility matches a user's sex.

    Returns None if user_sex is not provided or is unknown.
    """
    if not user_sex:
        return None
    u = str(user_sex).strip().lower()
    if u in {"male", "m", "1"}:
        u_norm = "MALE"
    elif u in {"female", "f", "2"}:
        u_norm = "FEMALE"
    else:
        return None

    t = (trial_sex or "").strip().upper()
    if not t or t == "ALL":
        return True
    return t == u_norm

def _extract_baseline_demographics(results_section: dict) -> dict:
    """Extract a lightweight baseline demographics view (if results are posted)."""
    baseline = results_section.get("baselineCharacteristicsModule") or {}
    measures = _as_list(baseline.get("measures"))
    keep_titles = {"age", "sex", "weight", "bmi", "body mass"}
    kept = []
    for m in measures:
        if not isinstance(m, dict):
            continue
        title = str(m.get("title") or "")
        t_lower = title.lower()
        if any(k in t_lower for k in keep_titles):
            kept.append(
                {
                    "title": m.get("title"),
                    "paramType": m.get("paramType"),
                    "unitOfMeasure": m.get("unitOfMeasure"),
                    "dispersionType": m.get("dispersionType"),
                    "classes": m.get("classes"),
                }
            )

    denoms = _as_list(baseline.get("denoms"))
    groups = _as_list(baseline.get("groups"))
    return {
        "has_results": bool(baseline),
        "groups": groups,
        "denoms": denoms,
        "measures": kept,
    }

def _extract_adverse_events(results_section: dict) -> dict:
    ae = results_section.get("adverseEventsModule") or {}
    return {
        "has_results": bool(ae),
        "frequencyThreshold": ae.get("frequencyThreshold"),
        "eventGroups": ae.get("eventGroups") or [],
        "seriousEvents": ae.get("seriousEvents") or [],
        "otherEvents": ae.get("otherEvents") or [],
    }

def _summarize_study(
    study: dict,
    *,
    user_age: Optional[int] = None,
    user_sex: Optional[str] = None,
    user_weight_kg: Optional[float] = None,
) -> dict:
    """Create a user-facing summary dict from a ClinicalTrials.gov v2 study record."""
    protocol = study.get("protocolSection") or {}
    results = study.get("resultsSection") or {}

    ident = protocol.get("identificationModule") or {}
    status = protocol.get("statusModule") or {}
    sponsors = protocol.get("sponsorCollaboratorsModule") or {}
    conditions_mod = protocol.get("conditionsModule") or {}
    design = protocol.get("designModule") or {}
    arms_mod = protocol.get("armsInterventionsModule") or {}
    eligibility = protocol.get("eligibilityModule") or {}
    outcomes = protocol.get("outcomesModule") or {}

    trial_sex = eligibility.get("sex")
    min_age_years = _parse_age_years(eligibility.get("minimumAge"))
    max_age_years = _parse_age_years(eligibility.get("maximumAge"))

    age_match = None
    if user_age is not None and min_age_years is not None and max_age_years is not None:
        try:
            a = int(user_age)
            age_match = (min_age_years <= a <= max_age_years)
        except (TypeError, ValueError):
            age_match = None

    sex_match = _sex_matches(trial_sex, user_sex)

    # Weight matching is rarely present in CT.gov structured fields; only evaluate
    # if baseline includes it (we expose baseline measures so the UI can decide).
    _ = user_weight_kg

    interventions = []
    for i in _as_list(arms_mod.get("interventions")):
        if not isinstance(i, dict):
            continue
        interventions.append(
            {
                "type": i.get("type"),
                "name": i.get("name"),
                "otherNames": i.get("otherNames") or [],
                "description": i.get("description"),
                "armGroupLabels": i.get("armGroupLabels") or [],
            }
        )

    arms = []
    for a in _as_list(arms_mod.get("armGroups")):
        if not isinstance(a, dict):
            continue
        arms.append(
            {
                "label": a.get("label"),
                "type": a.get("type"),
                "description": a.get("description"),
                "interventionNames": a.get("interventionNames") or [],
            }
        )

    posted_outcomes = _as_list((results.get("outcomeMeasuresModule") or {}).get("outcomeMeasures"))

    return {
        "nctId": ident.get("nctId"),
        "briefTitle": ident.get("briefTitle"),
        "officialTitle": ident.get("officialTitle"),
        "overallStatus": status.get("overallStatus"),
        "startDate": _deep_get(status, "startDateStruct.date"),
        "completionDate": _deep_get(status, "completionDateStruct.date"),
        "hasResults": bool(study.get("hasResults")),
        "sponsors": {
            "leadSponsor": sponsors.get("leadSponsor"),
            "collaborators": sponsors.get("collaborators") or [],
        },
        "conditions": {
            "conditions": conditions_mod.get("conditions") or [],
            "keywords": conditions_mod.get("keywords") or [],
        },
        "design": {
            "studyType": design.get("studyType"),
            "phases": design.get("phases") or [],
            "enrollment": design.get("enrollmentInfo"),
            "designInfo": design.get("designInfo"),
        },
        "interventions": interventions,
        "arms": arms,
        "eligibility": {
            "sex": trial_sex,
            "minimumAge": eligibility.get("minimumAge"),
            "maximumAge": eligibility.get("maximumAge"),
            "healthyVolunteers": eligibility.get("healthyVolunteers"),
            "criteria": eligibility.get("eligibilityCriteria"),
        },
        "applicability": {
            "userProvided": {
                "age": user_age,
                "sex": user_sex,
                "weightKg": user_weight_kg,
            },
            "ageInRange": age_match,
            "sexEligible": sex_match,
            "notes": [
                "Weight is often not available in structured eligibility; compare only if baseline measures report it.",
            ],
        },
        "outcomes": {
            "primaryOutcomes": outcomes.get("primaryOutcomes") or [],
            "secondaryOutcomes": outcomes.get("secondaryOutcomes") or [],
            "postedOutcomeMeasures": posted_outcomes,
        },
        "baselineDemographics": _extract_baseline_demographics(results),
        "adverseEvents": _extract_adverse_events(results),
    }

@mcp.tool()
def search_clinical_trials(
    conditions: Optional[list[str]] = None, 
    intervention: Optional[str] = None,
    max_results: int = 5,
    user_age: Optional[int] = None,
    user_sex: Optional[str] = None,
    user_weight_kg: Optional[float] = None,
) -> dict:
    """Search ClinicalTrials.gov and return user-oriented trial summaries.
    
    Args:
        conditions (Optional): Conditions/disease terms to search for.
        intervention (Optional): Intervention/drug term to search for.
        max_results (Optional): Maximum number of studies to retrieve from the first page (default: 5).
        user_age (Optional): Optional user age in years (used only for basic eligibility comparison).
        user_sex (Optional): Optional user sex (male/female/unknown or 1/2/0).
        user_weight_kg (Optional): Optional user weight in kg (CT.gov often doesn't provide comparable fields).
        
    Returns:
        A dict containing the query, a list of summarized studies, and pagination token.
    """
    
    # Define the API endpoint and parameters
    url = _CLINICAL_TRIALS_DOT_GOV_BASE_URL
    params = {}
    if conditions:
        params["query.cond"] = " AND ".join([c for c in conditions if c])
    if intervention:
        params["query.intr"] = intervention
    try:
        page_size = max(1, min(int(max_results), 50))
    except (TypeError, ValueError):
        page_size = 5
    params["pageSize"] = page_size
    params["sort"] = "@relevance"
    try: 
        resp = _HTTP_SESSION.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        studies = data.get("studies") or []
        summaries = [
            _summarize_study(
                s,
                user_age=user_age,
                user_sex=user_sex,
                user_weight_kg=user_weight_kg,
            )
            for s in studies
            if isinstance(s, dict)
        ]

        return {
            "query": {
                "conditions": conditions,
                "intervention": intervention,
                "pageSize": page_size,
                "sort": params.get("sort"),
            },
            "nextPageToken": data.get("nextPageToken"),
            "count": len(summaries),
            "studies": summaries,
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"API call failed: {str(e)}"}
    
@mcp.prompt()
def generate_system_prompt() -> str:
    return dedent(
        """ROLE
        You are a clinical-trial evidence collector supporting a patient-facing drug education system. Your sole mandate is to search for, gather, and condense study data relevant to the provided drug context and user profile. Do not produce reader-facing explanations or advice; downstream agents handle interpretation.

        CONTEXT YOU RECEIVE
        - Drug and condition cues from the user or orchestration layer.
        - Structured study summaries returned by the search tool (eligibility, outcomes, safety, identifiers).
        - Optional patient attributes such as age, sex, or treatment history for cohort matching.

        TOOLKIT
        - search_clinical_trials(conditions?, intervention?, max_results?, user_age?, user_sex?, user_weight_kg?): queries ClinicalTrials.gov and returns summarized studies. Invoke it whenever additional evidence is needed or filters change, and record when no relevant trials are found.

        WORKFLOW
        1. Confirm incoming drug context and user filters, then issue targeted tool calls to assemble an evidence pack covering eligibility, outcomes, and safety for comparable cohorts.
        2. Inspect returned studies for design details, population characteristics, efficacy outcomes, and adverse events. Focus on factual extraction, not interpretation.
        3. Discard registry boilerplate or metadata that does not affect safety, effectiveness, eligibility, or context.
        4. Condense findings into structured notes that preserve provenance (trial identifiers, registry references, data fields) so the explainer agent can cite them accurately.
        5. Flag data gaps, inconsistencies, or limitations that downstream agents must consider, without drawing conclusions for the user.

        ANALYSIS RULES
        - Keep observations at the evidence level; do not generalize beyond what the studies report.
        - Highlight how each trial aligns or differs from the provided patient profile (age bands, sex eligibility, condition focus) and state when similarity is approximate or unknown.
        - Separate efficacy endpoints, safety findings, and study design attributes so downstream agents can assemble tailored narratives.
        - Maintain professional boundaries: no clinical guidance, recommendations, or reassurances.
        - Preserve critical terminology from source records when it improves downstream traceability and citation.

        REQUIRED DISCLOSURES FOR THE PACK
        - Note that clinical trials enroll selected participants under controlled conditions and may not represent all patients.
        - Remind that trial results do not predict individual outcomes and may lack diversity in age, comorbidities, or concomitant treatments.

        OUTPUT TEMPLATE (DO NOT ADD EXPLANATORY NARRATIVE)
        1. Sources Queried: list tool calls, query parameters, and whether additional searches are pending.
        2. Cohort Summary: outline trial identifiers, enrollment scale, design type, and how each trial matches the provided profile.
        3. Evidence Pack Items:
           - Efficacy Signals: primary and secondary outcomes as reported, noting measurement methods.
           - Safety Signals: adverse events observed, grouped by seriousness and frequency when available.
           - Study Constraints: duration, comparator details, monitoring intensity, and missing subgroups.
        4. Gaps & Flags: highlight missing data, conflicting results, or follow-up actions for the explainer agent.

        TONE
        Factual, concise, and operational. Deliver machine-ready evidence notes and stop short of patient-facing explanation.
        """
    )

if __name__ == "__main__":
    mcp.run(transport='stdio')