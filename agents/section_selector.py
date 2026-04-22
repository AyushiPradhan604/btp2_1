from pydantic import BaseModel
from typing import List
from utils.llm_client import get_structured_completion

class SectionSelection(BaseModel):
    selected_sections: List[str]

def select_sections(paper_text: str) -> List[str]:
    """Selects the most relevant sections for a poster."""
    system_prompt = '''
You are an expert academic editor. Given the text or structure of a research paper, extract the names of the most important sections for a conference poster.
Rules:
- Select 5-7 sections max.
- Must include equivalents of: Title, Problem, Method, Results, Conclusion.
- Merge redundant sections (e.g., "Methodology" + "Approach").
- Ignore references, acknowledgements, appendices.
- If sections are missing, infer from content.
- If paper is unstructured, create logical section names.
'''
    import re
    headers = [line for line in paper_text.split('\n') if line.startswith('#')]
    structure = "\n".join(headers)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Paper Headers/Structure:\n{structure}\n\nSample Content:\n{paper_text[:2000]}"}
    ]
    
    result = get_structured_completion(messages, SectionSelection)
    if result:
        return result.selected_sections
    
    # Fallback
    return ["Introduction", "Methodology", "Results", "Discussion", "Conclusion"]
