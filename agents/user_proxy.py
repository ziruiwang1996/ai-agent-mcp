from ..context import Context

def user_proxy_agent(user_input):
    print(f"[UserProxyAgent] Received input: {user_input}")
    return Context(user_input)
