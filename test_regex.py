import re

def test():
    cases = [
        # LLM correctly escaped backslashes (should stay double)
        r'{"b": "\\sigma"}',
        
        # LLM output single backslash for macro (needs doubling)
        r'{"b": "\sigma"}',
        
        # LLM output single backslash for math macro that starts with a JSON escape char
        r'{"b": "\text"}',
        r'{"b": "\frac"}',
        
        # LLM output escaped quote (must stay escaped)
        r'{"b": "here is a \"quote\""}',
        
        # LLM correctly output double backslash for math macro that starts with JSON escape char
        r'{"b": "\\text"}',
    ]

    for s in cases:
        # Regex: match a backslash that is:
        # 1. NOT preceded by a backslash (?<!\\)
        # 2. NOT followed by double quote or another backslash (?!["\\/])
        res = re.sub(r'(?<!\\)\\(?!["\\/])', r'\\\\', s)
        print(f"Original: {s}")
        print(f"Fixed:    {res}")
        try:
            import json
            parsed = json.loads(res)
            print(f"Parsed:   {parsed['b']}")
        except Exception as e:
            print(f"Error:    {e}")
        print("-" * 40)

if __name__ == "__main__":
    test()
