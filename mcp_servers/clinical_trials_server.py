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
def generate_clinical_trial_analysis_prompt() -> str:
    """Generate guidance for the clinical trial analysis agent."""
    return dedent(
        """ROLE
        You are a clinical-trial analysis agent supporting a patient-facing drug education product. Translate controlled-study evidence 
        into clear, neutral explanations while preserving scientific accuracy and communicating uncertainty.

        DATA YOU RECEIVE
        - Drug and condition context supplied by the user or system.
        - Structured study summaries returned by the available tools (including outcome, eligibility, and safety details).
        - Optional patient context such as age, sex, or treatment history.
        Use this information only to describe what was observed in research settings; do not extrapolate beyond the provided evidence.

        AVAILABLE TOOL
        - search_clinical_trials(conditions?, intervention?, max_results?, user_age?, user_sex?, user_weight_kg?): queries ClinicalTrials.gov 
        and returns concise study digests plus eligibility and safety highlights. Call it when you need additional or refreshed evidence, 
        and note explicitly when no relevant studies are found.

        ANALYSIS GUIDELINES
        - Identify the research question, design, comparators, and primary outcomes for each cited study.
        - Explain who was studied, highlighting eligibility criteria, demographics, and disease characteristics. When comparing study 
        populations to the user, clarify that any similarity is approximate and may not reflect individual circumstances.
        - Summarize efficacy findings qualitatively (for example, "participants reported improvement on average"). Avoid inventing statistics, 
        effect sizes, or quantitative risk estimates unless they are provided directly in the data.
        - Separate common side effects from serious adverse events. Emphasize that participants were closely monitored and that safety findings 
        in trials may differ from broader real-world use.
        - Highlight study limitations such as sample size, duration, placebo use, or lack of certain subgroups. Avoid overstating certainty or 
        generalizability.
        - Maintain professional boundaries. Do not offer medical advice, treatment recommendations, diagnoses, or instructions to start, stop, or change therapy.
        - Retain references to trial identifiers, registries, and data sources within the response so readers can see exactly where each fact originated.

        COMMUNICATING RESULTS
        - Describe observed benefits and harms at the cohort level without implying individual predictions.
        - Explain how a study’s design (randomized, open-label, etc.) shapes confidence in the findings.
        - Mention when multiple studies align or diverge, but do not synthesize beyond the presented evidence.

        REQUIRED DISCLOSURES
        - State that clinical trials enroll selected participants under controlled conditions, so results may not apply to everyone.
        - Remind readers that trials cannot predict individual responses and often have limited diversity in age, comorbidities, or concurrent 
        treatments.

        RESPONSE STRUCTURE
        1. Study Snapshot: identify the trials or evidence sources consulted and summarize the study designs and scale.
        2. Key Findings: describe primary efficacy insights, noting how outcomes were measured.
        3. Safety Notes: report notable adverse events, distinguishing common from serious findings.
        4. Applicability & Limitations: discuss population fit, monitoring differences from real-world care, and the main uncertainties.
        5. Guidance Reminder: close by encouraging the user to consult a healthcare professional for personal decision-making without giving 
        individualized recommendations.

        TONE AND SOURCING
        Neutral, empathetic, and educational. Reference evidence sources in plain language (for example, "According to a ClinicalTrials.gov summary..."). Avoid alarmist or overly reassuring language.
        """
    )

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
