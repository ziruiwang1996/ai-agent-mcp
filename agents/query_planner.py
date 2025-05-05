from ..context import Context

def query_planner_agent(context:Context):
    print("[QueryPlannerAgent] Planning sub-questions...")
    goal = context.original_query
    # Simulated sub-questions
    subqs = [
        f"What KRAS inhibitors are under investigation in NSCLC?",
        f"What are the clinical trial results for KRAS inhibitors?",
        f"Which drugs have FDA approval for KRAS mutations in NSCLC?"
    ]
    for q in subqs:
        context.add_subquestion(q)
    return context
