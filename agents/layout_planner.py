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
    
    scores = {}
    total_score = 0
    for s in sections:
        name = s['name']
        if "title" in name.lower():
            layout_map[name] = {"col": -1} # Title ignores columns
            continue
        bullets_count = len(s.get('bullets', []))
        figures_count = len(s.get('figures', []))
        score = 50 + (bullets_count * 60) + (figures_count * 450)
        scores[name] = score
        
    col_scores = [0, 0, 0]
    
    for s in sections:
        name = s['name']
        if "title" in name.lower():
            layout_map[name] = {"col": -1}
            continue

        # Find column with lowest score
        min_col = col_scores.index(min(col_scores))
        layout_map[name] = {"col": min_col}
        col_scores[min_col] += scores[name]
        
    return layout_map
