import fitz
import os
import cv2
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)

def parse_pdf_structured(pdf_path: str, output_dir: str = "assets/figures") -> dict:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF missing: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError:
        logger.error("Missing Ultralytics or HF Hub. Ensure installed.")
        return {}

    print("Loading YOLOv8 DocLayNet model...")
    model_path = hf_hub_download(repo_id="pranavvdhawann/YOLOv8X-doclaynet", filename="model.pt")
    model = YOLO(model_path)
    
    doc = fitz.open(pdf_path)
    
    title = doc.metadata.get("title", "") if doc.metadata else ""
    authors = doc.metadata.get("author", "") if doc.metadata else ""
    
    structured_data = [] 
    figures_info = []
    
    img_index = 0
    ref_found = False
    
    for page_num in range(len(doc)):
        if ref_found:
            print("References reached. Stopping parser.")
            break
            
        print(f"Parsing Page {page_num+1}...")
        page = doc[page_num]
        scale = 200 / 72.0
        
        pix = page.get_pixmap(dpi=200)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
           img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
           
        results = model(img_array, verbose=False)
        yolo_boxes = []
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls[0].item())]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pdf_rect = fitz.Rect(x1/scale, y1/scale, x2/scale, y2/scale)
                yolo_boxes.append({
                    "class": class_name, 
                    "rect": pdf_rect,
                    "img_coords": (x1, y1, x2, y2)
                })
                
        # Consolidate internal subdivisions: Merge nearby or overlapping Picture boxes
        merged_pictures = []
        for yb in yolo_boxes:
            if yb['class'] == 'Picture':
                merged = False
                for mb in merged_pictures:
                    # Merge ONLY if bounding boxes heavily intersect to prevent grouping independent elements
                    if mb['rect'].intersects(yb['rect']):
                        mb['rect'] = mb['rect'] | yb['rect']
                        mx1 = int(min(mb['img_coords'][0], yb['img_coords'][0]))
                        my1 = int(min(mb['img_coords'][1], yb['img_coords'][1]))
                        mx2 = int(max(mb['img_coords'][2], yb['img_coords'][2]))
                        my2 = int(max(mb['img_coords'][3], yb['img_coords'][3]))
                        mb['img_coords'] = (mx1, my1, mx2, my2)
                        merged = True
                        break
                if not merged:
                    merged_pictures.append(yb)
        
        # Keep non-pictures and add merged pictures
        yolo_boxes = [yb for yb in yolo_boxes if yb['class'] != 'Picture'] + merged_pictures

                
        text_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        
        for b in text_blocks:
            if ref_found: break
            
            if b['type'] == 0:
                block_rect = fitz.Rect(b['bbox'])
                
                # Check for structural table natively before text parsing
                tables = page.find_tables(clip=block_rect)
                if tables and tables.tables:
                    parsed_table = tables.tables[0].extract()
                    structured_data.append({
                        "type": "Table",
                        "content": parsed_table,
                        "bbox": list(b['bbox']),
                        "page": page_num + 1
                    })
                    continue
                
                text_content = ""
                max_size = 0
                for l in b["lines"]:
                    for s in l["spans"]:
                        text_content += s["text"]
                        if s["size"] > max_size: max_size = s["size"]
                text_content = text_content.strip()
                if not text_content: continue
                
                if max_size > 11:
                     if not title: title = text_content
                     block_type = "Section-header"
                else:
                     block_type = "Text"
                
                if block_type == "Section-header":
                    lower_text = re.sub(r'[^a-zA-Z]', '', text_content.lower())
                    if lower_text in ["references", "bibliography", "acknowledgments", "acknowledgment"]:
                         ref_found = True
                         break
                         
                in_formula = False
                for yb in yolo_boxes:
                    if yb['class'] == 'Formula' and block_rect.intersects(yb['rect']):
                         block_type = "Formula"
                         # Encode formula explicitly inside the markdown context 
                         # (since we bypass docling for crash safety)
                         text_content = f"$$ {text_content} $$"
                         in_formula=True
                         break
                         
                if not in_formula and text_content.lower().startswith("fig"):
                    block_type = "Caption"
                    
                structured_data.append({
                    "type": block_type,
                    "content": text_content,
                    "bbox": list(b['bbox']),
                    "page": page_num + 1
                })
                
        for yb in yolo_boxes:
            if yb['class'] == 'Picture':
                pic_rect = yb['rect']
                caption_text = ""
                figure_num = ""
                best_dist = float('inf')
                
                # Identify caption and strictly chop matching coordinates out of YOLO's raw bounding box!
                caption_pdf_y0 = pic_rect.y1
                
                for b_obj in structured_data:
                    if b_obj['page'] == page_num + 1:
                        bx0, by0, bx1, by1 = b_obj['bbox']
                        # Look for text immediately underneath or inside the lower portion of the Picture box
                        if by0 >= pic_rect.y1 - 60 and by0 <= pic_rect.y1 + 150:
                            b_text = str(b_obj['content'])
                            if b_text.lower().startswith(('fig', 'table')):
                                dist = by0 - pic_rect.y1
                                if dist < best_dist:
                                    best_dist = dist
                                    caption_text = b_text
                                    caption_pdf_y0 = by0
                                    
                match = re.search(r'(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?)', str(caption_text), re.IGNORECASE)
                if match:
                    figure_num = match.group(0)
                else:
                    caption_text = f"Figure on Page {page_num+1}"
                    
                x1, y1, x2, y2 = yb['img_coords']
                # Subvert YOLO's mistake if it boxed the caption inside the graphic:
                caption_img_y0 = int(caption_pdf_y0 * scale)
                if caption_img_y0 < y2 and caption_img_y0 > y1 + 50:
                    y2 = caption_img_y0 - 2
                
                pad = 15
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(img_array.shape[1], x2 + pad), min(img_array.shape[0], y2 + pad)
                
                cropped_img = img_array[cy1:cy2, cx1:cx2]
                if cropped_img.shape[0] < 50 or cropped_img.shape[1] < 50:
                    continue
                    
                # Preserve complete figure boundaries natively (Constraint 7)
                cropped_tight = cropped_img
                
                img_index += 1
                img_filename = f"fig_p{page_num+1}_{img_index}.png"
                img_path = os.path.join(output_dir, img_filename).replace('\\', '/')
                cv2.imwrite(img_path, cv2.cvtColor(cropped_tight, cv2.COLOR_RGB2BGR))
                
                # Use strictly the corrected box
                final_rect = fitz.Rect(x1/scale, y1/scale, x2/scale, y2/scale)

                figures_info.append({
                    "image_path": os.path.abspath(img_path).replace('\\', '/'),
                    "caption": caption_text,
                    "figure_number": figure_num,
                    "bbox": list(final_rect),
                    "page": page_num + 1
                })

    try:
        import pymupdf4llm
        text_fallback = pymupdf4llm.to_markdown(doc)
    except Exception as e:
        print(f"[WARN] PyMuPDF4LLM failed ({e}), falling back to raw text extraction.")
        text_fallback = ""
        for b in structured_data:
            if b['type'] == 'Section-header':
                text_fallback += f"\n# {b['content']}\n"
            elif b['type'] == 'Formula':
                text_fallback += f"\n{b['content']}\n"
            elif b['type'] == 'Text':
                t = str(b['content'])
                # Natively transliterate PDF unicode symbols to LaTeX so LLM and HTML can render them!
                t = t.replace('\u2208', ' \\in ').replace('\u00d7', ' \\times ').replace('\u2211', ' \\sum ').replace('\u221e', ' \\infty ')
                t = t.replace('\u2264', ' \\leq ').replace('\u2265', ' \\geq ').replace('\u2192', ' \\rightarrow ').replace('\u2225', ' \\| ')
                t = t.replace('\u03bb', ' \\lambda ').replace('\u03c3', ' \\sigma ').replace('\u03f5', ' \\epsilon ').replace('\u03b1', ' \\alpha ')
                t = t.replace('\u03b2', ' \\beta ').replace('\u03b3', ' \\gamma ').replace('\u2212', ' - ').replace('\u2248', ' \\approx ')
                t = t.replace('\u2260', ' \\neq ').replace('\u00b7', ' \\cdot ').replace('\u22c6', ' \\star ').replace('\u22a4', ' \\top ')
                t = t.replace('\u2297', ' \\otimes ')
                
                text_fallback += t + " "
    # Mathematically lock figures to their originating layout section!
    current_section = "Introduction"
    for p_num in range(1, len(doc)+1):
        p_elements = [e for e in structured_data if e['page'] == p_num]
        p_figures = [f for f in figures_info if f['page'] == p_num]
        
        all_layout = sorted(p_elements + p_figures, key=lambda x: x['bbox'][1])
        for el in all_layout:
            if el.get('type') == 'Section-header':
                current_section = el['content']
            if 'image_path' in el:
                # Modifies object directly in figures_info array
                el['native_section'] = current_section
    
    if not title: title = "Research Poster"
    
    # --- BONUS: Draw bounding boxes natively onto the PDF and save it! ---
    annotated_path = pdf_path.replace(".pdf", "_annotated.pdf")
    for b in structured_data:
        try:
            page = doc[b['page'] - 1]
            color = (1, 0, 0) # Red for normal text
            if b['type'] == 'Section-header': color = (0, 0, 1) # Blue
            elif b['type'] == 'Table': color = (0, 1, 0) # Green
            elif b['type'] == 'Formula': color = (1, 0, 1) # Magenta
            elif b['type'] == 'Caption': color = (0, 1, 1) # Cyan
            
            page.draw_rect(fitz.Rect(b['bbox']), color=color, width=1.5)
        except Exception:
            pass
            
    for f in figures_info:
        try:
            page = doc[f['page'] - 1]
            # Orange for perfectly extracted images
            page.draw_rect(fitz.Rect(f['bbox']), color=(1, 0.5, 0), width=2.5) 
        except Exception:
            pass
            
    doc.save(annotated_path)
    print(f"\n[DEBUG] Saved fully annotated bounding-box visualizer to: {annotated_path}\n")
            
    res = {
        "title": title,
        "authors": authors,
        "content": text_fallback,
        "layout_tree": structured_data,
        "figures": figures_info
    }
    return res

if __name__ == "__main__":
    out = parse_pdf_structured("0031_Paper_41_paper.pdf", "assets/figures")
    print("\nExtraction Success!")
    print(f"Content Length: {len(out['content'])}")
    print(f"Figures extracted: {len(out['figures'])}")
    if len(out['figures']) > 0:
         print(f"Sample Figure: {out['figures'][0]}")
