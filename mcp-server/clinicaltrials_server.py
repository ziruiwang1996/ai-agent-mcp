import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ClinicalTrials")

@mcp.tool()
def search_clinical_trials(
    cond: str = None, 
    intr: str = None,
    ids: str = None,
    max_results: int = 5) -> dict:
    """
    Search for clinical trials based on queries and return the results.
    
    Args:
        cond: conditions or disease to search for
        intr: intervention or treatment to search for
        ids: study IDs to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        Dictionary containing the search results
    """
    
    # Define the API endpoint and parameters
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {}
    if cond:
        params["query.cond"] = cond
    if intr:
        params["query.intr"] = intr
    if ids:
        params["query.id"] = ids
    params["pageSize"] = max_results

    response = requests.get(url, params=params)
    print("Final URL:", response.url) 
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return {"error": f"Failed to fetch data: {response.status_code}"}

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
