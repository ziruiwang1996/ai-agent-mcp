import re
import os

def expand_env_in_text(text: str) -> str:
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")
    def repl(match):
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            raise KeyError(f"Environment variable '{var}' not set for configuration substitution")
        return val
    return pattern.sub(repl, text)