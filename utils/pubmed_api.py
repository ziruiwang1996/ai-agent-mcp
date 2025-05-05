# PubMed API querying and parsing helpers
import requests
from xml.etree import ElementTree

def search_pubmed(query, max_results=5):
    """Return a list of PubMed IDs matching the query."""
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "xml"}
    r = requests.get(esearch_url, params=params)
    tree = ElementTree.fromstring(r.content)
    return [id_elem.text for id_elem in tree.findall(".//IdList/Id")]

def fetch_pubmed_details(id_list):
    """Fetch title & abstract for each PubMed ID."""
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "retmode": "xml", "id": ",".join(id_list)}
    r = requests.get(efetch_url, params=params)
    tree = ElementTree.fromstring(r.content)
    papers = []
    for art in tree.findall(".//PubmedArticle"):
        pmid = art.find(".//PMID").text
        title = (art.find(".//ArticleTitle").text or "").strip()
        abstract = (art.find(".//Abstract/AbstractText").text or "").strip()
        papers.append({"id": pmid, "title": title, "abstract": abstract})
    return papers
