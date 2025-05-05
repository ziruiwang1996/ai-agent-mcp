# Summarizes retrieved literature
from ..utils.llm import huggingface_llm_query
from ..context import Context

def summarizer_agent(context:Context):
    print("[SummarizerAgent] Summarizing retrieved papers...")
    for paper in context.papers:
        prompt = (
            f"Summarize the following abstract:\n"
            f"Title: {paper['title']}\n"
            f"Abstract: {paper['abstract']}"
        )
        summary = huggingface_llm_query(prompt)
        context.summaries.append({"paper_id": paper["id"], "summary": summary})
    return context
