from pydantic import BaseModel
from typing import List
from utils.llm_client import get_structured_completion

class SectionSelection(BaseModel):
    selected_sections: List[str]

def select_sections(paper_text: str) -> List[str]:
    """Selects the most relevant sections for a poster."""
    system_prompt = '''
You are an expert academic editor. Given the headers of a research paper, select the most important sections for a conference poster.
Rules:
- Select 5-6 sections max.
- CRITICAL: You MUST output the EXACT, verbatim headings exactly as they appear in the provided structure list. NEVER invent, adapt, or paraphrase new names.
- Ensure the sections flow logically: Introduction -> Core Methodology -> Experiments/Results -> Conclusion.
- If a section header is extremely long, still output its exact string verbatim to preserve string matching algorithms.
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
