from mcp.server.fastmcp import FastMCP
from urllib.parse import urlparse
from typing import Any, Optional
from textwrap import dedent
import requests

mcp = FastMCP("shared_tool")

_ALLOWED_API_HOSTS = {
    "api.fda.gov",
    "wsearch.nlm.nih.gov",
    "www.dictionaryapi.com",
}

_OPENFDA_ENDPOINTS: dict[str, str] = {
    # Common openFDA endpoints used across agents.
    "drug_label": "https://api.fda.gov/drug/label.json",
    "drug_event": "https://api.fda.gov/drug/event.json",
}

@mcp.resource("file:///openfda/endpoints")
def openfda_endpoints() -> str:
    """Agent reference: known openFDA endpoints supported by this shared tool server."""
    return """
        # openFDA endpoints
        - drug_event: https://api.fda.gov/drug/event.json
        - drug_label: https://api.fda.gov/drug/label.json
        Use `openfda_search(endpoint=..., search=...)` for flexible queries.
    """

@mcp.tool()
def make_api_call(
    url: Optional[str] = None,
    params: Optional[dict] = None,
    headers: Optional[dict[str, str]] = None,
    timeout_s: int = 15,
) -> dict:
    """Make a safe GET request to an allowlisted public API.
    This is a *shared* utility for agents. It is intentionally conservative:
    - Only `https` URLs
    - Host allowlist to avoid SSRF / unexpected egress
    - Basic `limit` clamping for common public API patterns

    Args:
        url: Full API URL.
        params: Query parameters (dict).
        headers: Optional request headers.
        timeout_s: Request timeout seconds (default: 15).

    Returns:
        Parsed JSON response dict on success, or `{error: ...}`.
    """
    if not url:
        return {"error": "Missing required url parameter."}

    try:
        parsed = urlparse(url)
    except Exception:
        return {"error": "Invalid URL."}

    if parsed.scheme != "https" or not parsed.netloc:
        return {"error": "Only https URLs are allowed."}
    if parsed.hostname not in _ALLOWED_API_HOSTS:
        return {"error": f"Host not allowed: {parsed.hostname}"}

    safe_params: dict[str, Any] = dict(params or {})
    if "limit" in safe_params:
        try:
            safe_params["limit"] = max(1, min(int(safe_params["limit"]), 100))
        except (TypeError, ValueError):
            safe_params["limit"] = 10

    try:
        response = requests.get(url, params=safe_params, headers=headers, timeout=max(1, int(timeout_s)))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API call failed: {str(e)}"}


@mcp.tool()
def openfda_search(endpoint: str, search: str, limit: int = 5, skip: int = 0) -> dict:
    """Search an openFDA endpoint using a custom Lucene-style `search` query.
    Use this when an agent wants a customized query against openFDA, and already
    knows which endpoint to use (e.g. drug label vs adverse events).

    Args:
        endpoint: Name of the endpoint from `file:///openfda/endpoints`.
            Example: `drug_label` or `drug_event`.
        search: openFDA Lucene-like search string.
        limit: Number of records to return (1–100, default: 5).
        skip: Offset for pagination (default: 0).

    Returns:
        Dict with `endpoint`, `query`, `count`, `results` on success, or `{error: ...}`.
    """
    base_url = _OPENFDA_ENDPOINTS.get(str(endpoint).strip())
    if not base_url:
        return {"error": f"Unknown openFDA endpoint: {endpoint}. See file:///openfda/endpoints"}

    try:
        limit_int = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit_int = 5
    try:
        skip_int = max(0, int(skip))
    except (TypeError, ValueError):
        skip_int = 0

    params = {"search": str(search), "limit": limit_int}
    if skip_int:
        params["skip"] = skip_int

    data = make_api_call(base_url, params)
    if isinstance(data, dict) and "error" in data:
        return data

    return {
        "endpoint": endpoint,
        "query": params,
        "count": (data or {}).get("meta", {}).get("results", {}).get("total"),
        "results": (data or {}).get("results", []),
    }


@mcp.tool()
def get_word_count(text: str) -> int:
    """
    Get the word count of the provided text.

    Args:
        text: The input text to count words in.
        
    Returns:
        The number of words in the text.
    """
    return len(text.split())

@mcp.tool()
def get_drug_name(set_id:str) -> dict:
    """Get drug names (brand, generic, substance) from openFDA drug label endpoint using set_id.

    Args:
        set_id: The set_id of the drug label.

    Returns:
        A dict containing brand_name, generic_name, and substance_name lists, or an error message
    """
    params = {
        "search": f"set_id:{set_id}", "limit": 1}
    try: 
        response = requests.get(_OPENFDA_ENDPOINTS.get("drug_label"), params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if results:
            drug_names = results[0].get("openfda", {})
            return {
                "brand_name": drug_names.get("brand_name", []),
                "generic_name": drug_names.get("generic_name", []),
                "substance_name": drug_names.get("substance_name", [])
            }
        return []
    except requests.exceptions.RequestException as e:
        return {"error": f"API call failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport='stdio')