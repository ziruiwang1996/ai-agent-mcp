from rcsbapi.search import TextQuery, SeqSimilarityQuery
from rcsbapi.data import DataQuery
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("PDB")

@mcp.tool()
def search_pdb_ids(query: str) -> list:
    """
    Search for PDB ids using a text query.
    
    Args:
        query: The search query string.
        
    Returns:
        A list containing matched PDB ids.
    """
    tq = TextQuery(value=query)
    results = tq()
    return list(results)

@mcp.tool()
def extract_pdb_data(pdb_id: str) -> dict:
    """
    Extract PDB data for a given PDB id.
    
    Args:
        pdb_id: The PDB id to extract data for.
        
    Returns:
        A dictionary containing the extracted PDB data.
    """
    dq = DataQuery(
        input_type="entry",
        input_ids=[pdb_id],
        return_data_list=["struct", "exptl", "entity_poly"],
    )
    results = dq.exec()
    return results

@mcp.tool()
def search_similar_sequence(
    sequence: str, 
    evalue_cutoff: float = 0.1,
    identity_cutoff: float = 0.0,
    sequence_type: str = "protein",
    max_results: int = 5
    ) -> list:
    """
    Search for sequence PDB ids using a sequence similarity query.
    
    Args:
        sequence: protein or nucleotide sequence
        evalue_cutoff: upper cutoff for E-value (lower is more significant). Defaults to 0.1.
        identity_cutoff: lower cutoff for percent sequence match (0-1). Defaults to 0.
        sequence_type: type of biological sequence (“protein”, “dna”, “rna”). Defaults to “protein”.
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        A list containing PDB ids with matched sequence.
    """
    sq = SeqSimilarityQuery(
        value=sequence, 
        evalue_cutoff=evalue_cutoff,
        identity_cutoff=identity_cutoff,
        sequence_type=sequence_type
    )
    results = sq()
    res = []

    for r in results:
        res.append(r)
        max_results -= 1
        if max_results == 0:
            break
    return res

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')