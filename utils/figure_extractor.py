import fitz
import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_figures(pdf_path: str, output_dir: str = "assets/figures") -> list:
    """Extracts images from PDF using YOLOv8 DocLayNet model."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF missing: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError:
         logger.error("Missing dependencies. Run `pip install huggingface_hub ultralytics opencv-python-headless`.")
         return []
         
    # Download and load YOLO model
    print("Loading YOLOv8 DocLayNet model (this will use cached weights if already downloaded)...")
    model_path = hf_hub_download(repo_id="pranavvdhawann/YOLOv8X-doclaynet", filename="model.pt")
    model = YOLO(model_path)
    
    # Target classes we want to save
    target_class_names = ["Picture", "Table"] 
    
    doc = fitz.open(pdf_path)
    figures_info = []
    
    img_index = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Render page to an image
        # High DPI for clearer cropped figures
        pix = page.get_pixmap(dpi=200) 
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to RGB (OpenCV expects BGR so we will eventually convert)
        if pix.n == 4:
           img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
           
        # PyMuPDF text blocks for spatial caption matching
        scale = 200 / 72.0
        text_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
           
        # Run YOLO inference
        results = model(img_array, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                
                # Check if this box is a picture/table
                if class_name in target_class_names:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crop image padding slightly
                    pad = 10
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(img_array.shape[1], x2 + pad)
                    y2 = min(img_array.shape[0], y2 + pad)
                    
                    cropped_img = img_array[y1:y2, x1:x2]
                    
                    # Ignore tiny spurious boxes
                    if cropped_img.shape[0] < 50 or cropped_img.shape[1] < 50:
                        continue
                    
                    img_index += 1
                    img_filename = f"fig_p{page_num+1}_{img_index}.png"
                    img_path = os.path.join(output_dir, img_filename).replace('\\', '/')
                    
                    # Convert RGB to BGR before writing
                    cv2.imwrite(img_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                    
                    # Spatial Caption Matching
                    pdf_y2 = y2 / scale
                    pdf_x1 = x1 / scale
                    pdf_x2 = x2 / scale
                    
                    caption_text = ""
                    best_dist = float('inf')
                    for b in text_blocks:
                        if b['type'] == 0:  # text block
                            b_x0, b_y0, b_x1, b_y1 = b['bbox']
                            # Check if block is roughly below the image and horizontally aligns
                            if b_y0 >= pdf_y2 - 20 and b_y0 <= pdf_y2 + 250:
                                if b_x1 >= pdf_x1 - 100 and b_x0 <= pdf_x2 + 100:
                                    b_text = ""
                                    for l in b["lines"]:
                                        for s in l["spans"]:
                                            b_text += s["text"]
                                    b_text = b_text.strip()
                                    if b_text.lower().startswith(('fig', 'table')):
                                        dist = b_y0 - pdf_y2
                                        if dist < best_dist:
                                            best_dist = dist
                                            caption_text = b_text

                    # Default fallback caption to aid generic routing if no text is found
                    if not caption_text:
                         caption_text = f"Generic {class_name} from Page {page_num+1}"
                    
                    figures_info.append({
                        "image_path": os.path.abspath(img_path).replace('\\', '/'),
                        "caption": caption_text
                    })

    return figures_info
