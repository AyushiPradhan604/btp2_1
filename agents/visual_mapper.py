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
You are a highly intelligent layout mapper. You must review available figures and strictly integrate them into sections only when profoundly applicable.
Rules:
- IMPORTANT CONTEXT-AWARE MAPPING: Only include an image if its specific caption (e.g., "Figure X") is explicitly referenced or inherently mathematically required by the section's text constraints.
- DO NOT arbitarily force image placement if the explicit context or caption title doesn't logically align perfectly with the section content.
- Ensure each figure is mapped EXACTLY ONCE globally. Do not duplicate figures.
- Select 1 to 2 relevant figures per section to avoid crowding.
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
                    # Enforce that the real PDF caption is used as a fallback if the LLM mangles it
                    caption_to_use = f.caption if len(f.caption) > 3 else af.get('caption', f"Figure for {section_name}")
                    mapped.append({"image_path": af['image_path'], "caption": caption_to_use})
                    break
        return mapped
    return []
