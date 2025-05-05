# Searches PubMed or Semantic Scholar API
from ..utils.pubmed_api import search_pubmed, fetch_pubmed_details
from ..context import Context

def literature_search_agent(context:Context, max_results=3):
    print("[LiteratureSearchAgent] Searching literature for each sub-question...")
    for subq in context.subquestions:
        if subq["status"] == "pending":
            ids = search_pubmed(subq["question"], max_results=max_results)
            papers = fetch_pubmed_details(ids)
            for paper in papers:
                context.add_paper(paper)
    return context
