from pydantic import BaseModel
from typing import List
from utils.llm_client import get_structured_completion

class ContentCompression(BaseModel):
    bullets: List[str]

def extract_section_text(section_name: str, paper_text: str) -> str:
    lines = paper_text.split('\n')
    section_content = []
    in_section = False
    
    # Strip non-alpha characters for robust matching
    sec_clean = ''.join(e for e in section_name.lower() if e.isalnum())
    
    for line in lines:
        if line.startswith('#'):
            h_clean = ''.join(e for e in line.lower() if e.isalnum())
            if sec_clean and (sec_clean in h_clean or h_clean in sec_clean):
                in_section = True
                continue
            elif in_section:
                break
        if in_section:
            section_content.append(line)
            
    res = "\n".join(section_content).strip()
    return res if res else paper_text[:4000] # Fallback if header not found

def compress_content(section_name: str, section_content: str) -> List[str]:
    """Converts section text into concise, poster-friendly bullet points."""
    system_prompt = '''
You are an expert science communicator formatting text for a highly visual academic poster.
Rules:
- Summarize the section accurately using 3 to 4 detailed bullet points.
- Aim for 15-20 words per bullet to ensure the text fits perfectly inside the restricted section box.
- Extract substantive insights, theoretical derivations, methodology details, and numerical results where available.
- Never output blank lines or "Summary not available" if content is present.
- Preserve mathematical equations and notation natively using LaTeX formatting ($ for inline, $$ for block). Do NOT strip out math.
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
