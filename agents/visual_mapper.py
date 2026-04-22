from pydantic import BaseModel
from typing import List, Dict
from utils.llm_client import get_structured_completion

class Figure(BaseModel):
    image_path: str
    caption: str

class SectionFigures(BaseModel):
    figures: List[Figure]

def map_figures(section_name: str, section_bullets: List[str], available_figures: List[Dict]) -> List[Dict]:
    """Match figures to the most relevant sections."""
    if not available_figures:
        return []
        
    system_prompt = '''
You are a highly intelligent layout mapper. You must review available figures and integrate them into the correct academic section to create a balanced poster.
Rules:
- Select 1 to 2 figures per section whenever available to prevent overflow. Do not overcrowd a single section.
- Distribute images evenly across sections where relevant. The final poster should have 5 to 6 images in total across all sections.
- Method -> system architectures or logic flows.
- Results -> plots or outcome matrices.
- Introduction -> generic overviews.
- Only select figures that strongly relate to the section text.
- Only provide EXACT image paths from the available list.
'''
    content = f"Section: {section_name}\nBullets:\n" + "\n".join(section_bullets)
    content += "\n\nAvailable Figures:\n"
    for fig in available_figures:
        content += f"- Path: {fig['image_path']}\n  Caption: {fig['caption']}\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]
    
    result = get_structured_completion(messages, SectionFigures)
    mapped = []
    if result:
        import os
        for f in result.figures:
            # Verify the output paths are actually in available_figures by basename
            base_f = os.path.basename(f.image_path)
            for af in available_figures:
                if os.path.basename(af['image_path']) == base_f:
                    mapped.append({"image_path": af['image_path'], "caption": f.caption})
                    break
        return mapped
    return []
