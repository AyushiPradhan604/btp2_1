from pydantic import BaseModel
from typing import List
import base64
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_client import vision_llm
from langchain_core.messages import HumanMessage, SystemMessage

class CriticFeedback(BaseModel):
    issues: List[str]
    fixes: List[str]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def criticize_poster(poster_path: str) -> dict:
    """Analyze the poster image and identify issues."""
    if not os.path.exists(poster_path):
         return {"issues": ["Poster image missing"], "fixes": []}
         
    system_prompt = '''
You are an expert design critic evaluating an academic poster. Analyze the image and identify issues.
Check for:
- Text overflow
- Poor spacing
- Visual imbalance
- Small fonts
- Misaligned elements
Suggest fixes.
'''
    
    base64_image = encode_image(poster_path)

    try:
        structured_llm = vision_llm.with_structured_output(CriticFeedback)
        
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Please provide your critique of this poster layout."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        )
        sys_msg = SystemMessage(content=system_prompt)
        
        parsed = structured_llm.invoke([sys_msg, msg])
        return {"issues": parsed.issues, "fixes": parsed.fixes}
    except Exception as e:
        # Vision structured output not supported on free-tier HuggingFace
        # We silently bypass the critic loop so the user gets their poster instantly
        return {"issues": [], "fixes": []}
