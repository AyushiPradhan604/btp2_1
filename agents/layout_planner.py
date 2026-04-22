from pydantic import BaseModel
from typing import List
from utils.llm_client import get_structured_completion

class Layout(BaseModel):
    x: float
    y: float
    w: float
    h: float

class SectionLayout(BaseModel):
    section_name: str
    layout: Layout

class PosterLayout(BaseModel):
    layouts: List[SectionLayout]

def plan_layout(sections: list) -> dict:
    """Assign sections to specific columns (0, 1, 2) mathematically for HTML Grid injection."""
    layout_map = {}
    
    col_content_score = [0, 0, 0]
    
    for s in sections:
        name = s['name']
        if "title" in name.lower():
            layout_map[name] = {"col": -1} # Title ignores columns
            continue
            
        bullets_count = len(s.get('bullets', []))
        figures_count = len(s.get('figures', []))
        
        # Heuristic "weight" of the section to balance columns
        score = 50 + (bullets_count * 60) + (figures_count * 450)
        
        shortest_col_idx = col_content_score.index(min(col_content_score))
        
        layout_map[name] = {"col": shortest_col_idx}
        col_content_score[shortest_col_idx] += score
        
    return layout_map
