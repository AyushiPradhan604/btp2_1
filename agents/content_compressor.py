from pydantic import BaseModel
from typing import List
from utils.llm_client import get_structured_completion

class ContentCompression(BaseModel):
    bullets: List[str]

def extract_section_text(section_name: str, paper_text: str) -> str:
    lines = paper_text.split('\n')
    section_content = []
    in_section = False
    
    import re
    # Strip numbers from the requested section name for cleaner matching
    clean_req_name = re.sub(r'^\d+(\.\d+)*\s*[\.\-]?\s*', '', section_name)
    sec_clean = ''.join(e for e in clean_req_name.lower() if e.isalnum())
    
    for line in lines:
        if line.startswith('#'):
            # Strip markdown and numbers from the actual header
            clean_h_name = re.sub(r'^\#+\s*\d+(\.\d+)*\s*[\.\-]?\s*', '', line)
            h_clean = ''.join(e for e in clean_h_name.lower() if e.isalnum())
            
            # Check if this header matches the target section
            is_match = False
            if sec_clean and h_clean:
                # Strong exact match or major substring match to prevent false positives
                if sec_clean == h_clean or (len(h_clean) > 5 and h_clean in sec_clean) or (len(sec_clean) > 5 and sec_clean in h_clean):
                    is_match = True
            
            if is_match:
                in_section = True
                continue
            elif in_section:
                # We hit a NEW header while already parsing our target section! STOP!
                break
        if in_section:
            section_content.append(line)
            
    res = "\n".join(section_content).strip()
    return res if res else paper_text[:4000] # Fallback if header not found

def compress_content(section_name: str, section_content: str) -> List[str]:
    """Converts section text into concise, poster-friendly bullet points."""
    system_prompt = '''
You are an expert science communicator formatting text for a highly visual academic poster.
Your goal is to extract deep technical information while crafting a compelling "mini-story".
Rules:
- Intelligently appraise and decide the priority of all available content. Do not oversimplify sections into generic bullet points. Re-synthesize the core essence and direct contributions of each section authentically without losing technical weight.
- Filter and prioritize key mathematical equations, rules, formulas, and parameters natively leveraging LaTeX formatting (e.g. `$`, `$$`). Strategically include these only if they form the structural anchor of the methodology or results.
- Narrative & Hook: Craft a strong central narrative thread (Problem -> Idea -> Why it matters). Include a strong "hook" to grab attention while preserving empirical rigor.
- "Aha Moment": Clearly articulate what is novel, parameter setups, or critical outcomes, articulating exactly why this is different from prior constraints natively.
- Summarize the section comprehensively using 5 to 7 meticulously dense bullet points (approx 25-40 words each) to preserve visual layout bounds and absolutely eliminate poster whitespace.
- Never output blank lines or "Summary not available".
'''
    actual_content = extract_section_text(section_name, section_content)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Section Name: {section_name}\n\nContent:\n{actual_content[:4000]}"}
    ]
    
    result = get_structured_completion(messages, ContentCompression)
    if result:
        return result.bullets
    return ["Summary not available."]
