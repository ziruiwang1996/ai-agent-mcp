from mcp.server.fastmcp import FastMCP
import requests
import os

mcp = FastMCP("adverse_event")

_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
_FAERS_SEARCHABLE_FIELDS_PATH = os.path.join(
    _ASSETS_DIR, "drug_adverse_event_fields.yaml"
)
_FAERS_BASE_URL = "https://api.fda.gov/drug/event.json"

@mcp.resource("file:///openfda/faers_searchable_fields")
def get_searchable_fields_of_FAERS():
    """Return FDA Adverse Event Reporting System (FAERS) searchable field documentation (YAML).

    This is intended to be used by the agent as a reference for building valid
    `search` queries against https://api.fda.gov/drug/event.json.
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

@mcp.tool()
def get_faers_reports(drug_name: str, limit: int = 5):
    """Fetch recent FDA Adverse Event Reporting System (FAERS) adverse event reports for a drug from openFDA.

    Args:
        drug_name: Drug name to search for (matches `medicinalproduct` field).
        limit: Number of reports to return (default: 5).

    Returns:
        A dict with `results` (list) on success, or `error` on failure.
    """
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "limit": max(1, min(int(limit), 100)),
    }
    try:
        resp = requests.get(_FAERS_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return {
            "drug_name": drug_name,
            "count": data.get("meta", {}).get("results", {}).get("total"),
            "results": data.get("results", []),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve FAERS reports: {str(e)}"}

@mcp.tool()
def get_adverse_event_with_demographics(set_id: str, age: int, weight: float, sex: str):
    """Fetch recent FAERS adverse event reports filtered by demographics.

    Uses openFDA drug/event endpoint with a `search` query.

    Args:
        set_id: SPL set ID to match against `patient.drug.openfda.spl_set_id`.
        age: Approximate onset age in years.
        weight: Approximate patient weight in kilograms.
        sex: Patient sex as "male"/"female"/"unknown" or "1"/"2"/"0".

    Returns:
        A dict with `query`, `count`, `reactions` on success, or `error` on failure.
    """
    sex_map = {
        "male": "1",
        "m": "1",
        "female": "2",
        "f": "2",
        "unknown": "0",
        "u": "0",
    }
    normalized_sex = sex_map.get(str(sex).strip().lower())
    if normalized_sex is None:
        return {"error": "Invalid sex; use male/female/unknown or 1/2/0."}

    try:
        age_val = int(age)
        weight_val = float(weight)
    except (TypeError, ValueError):
        return {"error": "Invalid age/weight; age must be int, weight must be number."}
    if age_val < 0 or weight_val <= 0:
        return {"error": "Invalid age/weight; age must be >= 0 and weight > 0."}

    age_window = 5
    weight_window = 5.0
    age_min = max(0, age_val - age_window)
    age_max = age_val + age_window
    weight_min = max(0.0, weight_val - weight_window)
    weight_max = weight_val + weight_window

    # Let requests handle URL encoding; build the Lucene query in plain form.
    query_terms = [
        f'patient.drug.openfda.spl_set_id:"{set_id}"',
        f'patient.patientonsetage:[{age_min} TO {age_max}]',
        'patient.patientonsetageunit:801',
        f'patient.patientweight:[{weight_min} TO {weight_max}]',
        f'patient.patientsex:{normalized_sex}',
    ]
    search_query = " AND ".join(query_terms)

    params = {
        "search": search_query
    }

    try:
        resp = requests.get(_FAERS_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        reactions: list[dict] = []
        for report in data.get("results", []) or []:
            patient = report.get("patient") or {}
            for reaction in patient.get("reaction", []) or []:
                if not isinstance(reaction, dict):
                    continue
                reactions.append(
                    {
                        "reactionmeddrapt": reaction.get("reactionmeddrapt"),
                        "reactionmeddraversionpt": reaction.get("reactionmeddraversionpt"),
                        "reactionoutcome": reaction.get("reactionoutcome"),
                    }
                )

        return {
            "query": {"search": search_query},
            "count": data.get("meta", {}).get("results", {}).get("total"),
            "reactions": reactions,
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve FAERS reports: {str(e)}"}

def average_days_adverse_event_after_medication():
    pass

if __name__ == "__main__":
    mcp.run(transport="stdio")
    #print(get_adverse_event_with_demographics("595f437d-2729-40bb-9c62-c8ece1f82780", 30, 70.0, "male"))
