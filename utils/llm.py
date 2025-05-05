# Abstraction layer for LLM calls (OpenAI, Claude, etc.)

import os
import openai

# Set your OPENAI_API_KEY in env before running
openai.api_key = os.getenv("OPENAI_API_KEY")

def huggingface_llm_query(prompt, model="gpt-4o-mini", max_tokens=500):
    pass

def openai_llm_query(prompt, model="gpt-4o-mini", max_tokens=500):
    """Send prompt to LLM and return the text response."""
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()
