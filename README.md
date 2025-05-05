# AI Research Assistant for Life Sciences #

The system uses agents to:
- Understand the question
- Retrieve and summarize scientific literature
- Track findings in a shared context
- Optionally suggest next experiments or related targets

Project Structure
```
ai_agent/
├── agents/
│   ├── user_proxy.py              # Receives user input and updates context
│   ├── query_planner.py           # Breaks queries into sub-questions
│   ├── literature_search.py       # Queries PubMed or Semantic Scholar
│   ├── summarizer.py              # Summarizes abstracts/full texts
│   └── hypothesis_generator.py    # Suggests new ideas from results
│
├── utils/
│   ├── pubmed_api.py              # PubMed API interface
│   └── llm.py                     # LLM abstraction layer (OpenAI, etc.)
│
├── context.py                     # ModelContextProtocol: shared BioContext class
├── main.py                        # Main script to orchestrate agents
├── examples/
│   └── sample_run.py              # Simple demo using a sample input
```

Agents & Context Flow
| Agent                        | Role         | Description                                                               |
| ---------------------------- | ------------ | ------------------------------------------------------------------------- |
| **UserProxyAgent**           | Interface    | Receives user query and adds it to context                                |
| **QueryPlannerAgent**        | Task planner | Breaks query into sub-questions (e.g., mechanism, clinical trials, drugs) |
| **LiteratureSearchAgent**    | Retriever    | Searches PubMed or Semantic Scholar via API                               |
| **SummarizerAgent**          | Synthesizer  | Extracts insights from papers                                             |
| **HypothesisGeneratorAgent** | Analyst      | (Optional) Proposes new experiments or connections                        |
| **MCP Context**              | Shared state | Holds tasks, results, memory, and logs                                    |
