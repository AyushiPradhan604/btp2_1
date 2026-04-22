import argparse
import os
import sys

# Ensure imports work from project root
sys.path.append(os.path.dirname(__file__))

from utils.pdf_parser import parse_pdf
from utils.figure_extractor import extract_figures
from agents.section_selector import select_sections
from agents.content_compressor import compress_content
from agents.visual_mapper import map_figures
from agents.layout_planner import plan_layout
from agents.renderer import render_poster
from agents.critic import criticize_poster

def main():
    parser = argparse.ArgumentParser(description="Poster Agent Pipeline")
    parser.add_argument("pdf_path", type=str, help="Path to the research paper PDF")
    parser.add_argument("--out", type=str, default="poster.png", help="Output poster image path")
    args = parser.parse_args()
    
    pdf_path = os.path.abspath(args.pdf_path)
    output_image = os.path.abspath(args.out)
    
    if not os.path.exists(pdf_path):
         print(f"Error: Could not find PDF at {pdf_path}")
         return
    
    # 1. Parse PDF
    print(f"1. Parsing PDF: {pdf_path}...")
    res = parse_pdf(pdf_path)
    paper_title = res.get("title", "Research Poster")
    paper_authors = res.get("authors", "")
    paper_content = res.get("content", "")
    
    print("2. Extracting Figures...")
    figures_dir = os.path.join(os.path.dirname(__file__), "assets", "figures")
    available_figures = extract_figures(pdf_path, output_dir=figures_dir)
    print(f"   Extracted {len(available_figures)} figures.")
    
    print("3. Selecting Sections...")
    selected_section_names = select_sections(paper_content)
    
    # Ensure Title is present
    if not any("title" in name.lower() for name in selected_section_names):
        selected_section_names.insert(0, "Title")
        
    sections_data = []
    
    print("4 & 5. Compressing Content and Mapping Figures...")
    for idx, sec_name in enumerate(selected_section_names):
        print(f"   Analysing section: {sec_name}")
        
        if "title" in sec_name.lower():
            bullets = []
            sec_figures = []
        else:
            bullets = compress_content(sec_name, paper_content)
            sec_figures = map_figures(sec_name, bullets, available_figures)
            
            # Ensure uniqueness by removing used figures from the available pool
            used_paths = {fig['image_path'] for fig in sec_figures}
            available_figures = [f for f in available_figures if f['image_path'] not in used_paths]
            
        sections_data.append({
            "name": sec_name,
            "priority": idx + 1,
            "bullets": bullets,
            "figures": sec_figures
        })
        
    print("6. Planning Layout...")
    layout_map = plan_layout(sections_data)
    
    # Inject layout into sections data
    for sec in sections_data:
        sec["layout"] = layout_map.get(sec["name"], {"x": 0.0, "y": 0.0, "w": 0.33, "h": 0.3})
        
    # Ensure all image paths are absolute for rendering
    for sec in sections_data:
         for fig in sec["figures"]:
              fig["image_path"] = os.path.abspath(fig["image_path"]).replace('\\', '/')
              
    data_contract = {
        "paper_title": paper_title,
        "paper_authors": paper_authors,
        "sections": sections_data
    }
    
    print("7. Initializing Render & Critic Loop...")
    critic_css = ""
    
    for iteration in range(3):
        print(f"\n--- [Iteration {iteration + 1} / 3] ---")
        success = render_poster(data_contract, output_image, critic_css=critic_css)
        if not success:
            print("Render failed. Aborting.")
            break
            
        print("   Poster rendered. Running Critic...")
        critic_feedback = criticize_poster(output_image)
        
        issues = critic_feedback.get("issues", [])
        fixes = critic_feedback.get("fixes", [])
        
        if not issues or (len(issues) == 1 and "missing" in issues[0].lower()):
            print("   No major issues found. Loop complete!")
            break
            
        print(f"   Critic Issues: {issues}")
        print(f"   Applying fixes CSS...")
        critic_css += "\n" + generate_css_fixes(fixes)
        
    print(f"\n[SUCCESS] Poster Agent processing complete! Final poster saved to: {output_image}")

def generate_css_fixes(fixes):
    """Invokes LLM to convert semantic fixes into pure CSS overrides."""
    from utils.llm_client import get_structured_completion
    from pydantic import BaseModel
    
    class CSSFix(BaseModel):
        css: str

    try:
         system_prompt = "You are a CSS expert. Given the design fixes, output pure CSS rules to override existing styles for classes like .section-box, .section-title, .bullet-list, .title-block, .figure-img, .poster-container. Adjust paddings, font sizes, margins etc. as requested. Output valid CSS only."
         
         messages = [
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": f"Please apply these fixes using CSS:\n{fixes}"}
         ]
         
         result = get_structured_completion(messages, CSSFix)
         return result.css if result else ""
    except Exception as e:
         return ""

if __name__ == "__main__":
    main()
