import argparse
import os
import sys

# Ensure imports work from project root
sys.path.append(os.path.dirname(__file__))

from utils.unified_parser import parse_pdf_structured
from agents.section_selector import select_sections
from agents.content_compressor import compress_content, extract_section_text
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
    
    # 1 & 2. Parse PDF and Extract Figures Structurally
    print(f"1 & 2. Parsing PDF Structurally: {pdf_path}...")
    figures_dir = os.path.join(os.path.dirname(__file__), "assets", "figures")
    res = parse_pdf_structured(pdf_path, output_dir=figures_dir)
    
    paper_title = res.get("title", "Research Poster")
    paper_authors = res.get("authors", "")
    paper_content = res.get("content", "")
    available_figures = res.get("figures", [])
    layout_tree = res.get("layout_tree", [])
    
    # Save the spatial tree with bounding boxes to a local file for inspection!
    import json
    with open("layout_dump.json", "w", encoding="utf-8") as f:
        json.dump(layout_tree, f, indent=4)
        
    print(f"   Extracted {len(available_figures)} tight figures, and fully mapped spatial table/equation layout tree (Saved to layout_dump.json).")
    
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
            
            # --- STRICT NATIVE MAPPING (Constraint 5 & 6) ---
            raw_sec_text = extract_section_text(sec_name, paper_content)
            sec_clean = ''.join(e for e in sec_name.lower() if e.isalnum())
            sec_figures = []
            
            for fig in available_figures:
                fig_num_str = str(fig.get('figure_number', '')).lower()
                is_referred = False
                
                # Context-Aware Check: Is "Figure X" physically mentioned in this section's raw text?
                if fig_num_str and len(fig_num_str) > 3:
                     # sometimes figure_number is "figure 1", sometimes "fig 1"
                     clean_fig_num = ''.join(e for e in fig_num_str if e.isalnum())
                     clean_raw_text = ''.join(e for e in raw_sec_text.lower() if e.isalnum())
                     if clean_fig_num in clean_raw_text:
                         is_referred = True
                         
                nat_clean = ''.join(e for e in fig.get('native_section', '').lower() if e.isalnum())
                
                # Assign if explicitly referenced in text, OR natively resides in this identically tracked section header layout
                if is_referred or (nat_clean and (nat_clean in sec_clean or sec_clean in nat_clean)):
                    sec_figures.append({"image_path": fig["image_path"], "caption": fig["caption"]})
            
            # Ensure absolute uniqueness by removing used figures from the pool globally
            used_paths = {fig['image_path'] for fig in sec_figures}
            available_figures = [f for f in available_figures if f['image_path'] not in used_paths]
            
        import re
        clean_name = re.sub(r'^\d+(\.\d+)*\s*[\.\-]?\s*', '', sec_name).strip()
        
        sections_data.append({
            "name": clean_name,
            "priority": idx + 1,
            "bullets": bullets,
            "figures": sec_figures
        })
        
    # --- DYNAMIC IMAGE PADDING TO PREVENT WHITESPACE ---
    total_imgs_mapped = sum(len(sec.get("figures", [])) for sec in sections_data)
    if total_imgs_mapped < 6 and available_figures:
        padding_needed = min(len(available_figures), 8 - total_imgs_mapped)
        for i in range(padding_needed):
            # Mathematically assign images iteratively moving backwards through sections 
            target_idx = -( (i % len(sections_data)) + 1 )
            if len(sections_data) >= abs(target_idx):
                 if "figures" not in sections_data[target_idx]:
                     sections_data[target_idx]["figures"] = []
                 sections_data[target_idx]["figures"].append({
                      "image_path": available_figures[i]["image_path"], 
                      "caption": available_figures[i]["caption"]
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
