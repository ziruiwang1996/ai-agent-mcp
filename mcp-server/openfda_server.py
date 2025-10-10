import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("openFDA")

@mcp.tool()
def search_drug(drug_name: str) -> dict:
    """
    Search for drug information using the OpenFDA API.
    
    Args:
        drug_name: The name of the drug to search for.
        
    Returns:
        A dictionary containing the drug information.
    """
    
    url = "https://api.fda.gov/drug/label.json"
    params = {
        "search": f"openfda.brand_name:{drug_name}",
        "limit": 1
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("results"):
            return data["results"][0]
        else:
            return {"error": "No results found."}
    else:
        params = {
            "search": f"openfda.generic_name:{drug_name}",
            "limit": 1
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                return data["results"][0]
            else:
                return {"error": "No results found."}
        else:
            return {"error": f"Request failed with status code {response.status_code}."}
    
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')