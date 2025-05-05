# Suggests new ideas based on findings
from ..utils.llm import openai_llm_query
from ..context import Context

def hypothesis_generator_agent(context:Context):
    print("[HypothesisGeneratorAgent] Generating hypotheses based on summaries...")
    combined = "\n".join([s["summary"] for s in context.summaries])
    prompt = (
        "Based on these summaries, propose potential hypotheses or "
        "next experimental directions:\n" + combined
    )
    hypothesis = openai_llm_query(prompt)
    context.add_hypothesis(hypothesis)
    return context
