# Sample script to run the whole pipeline
#!/usr/bin/env python3
from ..agents.user_proxy import user_proxy_agent
from ..agents.query_planner import query_planner_agent
from ..agents.literature_search import literature_search_agent
from ..agents.summarizer import summarizer_agent
from ..agents.hypothesis_generator import hypothesis_generator_agent

def main():
    user_input = input("Enter your biomedical query: ")
    context = user_proxy_agent(user_input)
    context = query_planner_agent(context)
    context = literature_search_agent(context)
    context = summarizer_agent(context)
    context = hypothesis_generator_agent(context)

    print("\n===== Final Context =====")
    print(context)

if __name__ == "__main__":
    main()
