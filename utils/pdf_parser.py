import fitz # PyMuPDF
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_pdf(pdf_path: str) -> dict:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF missing: {pdf_path}")
    
    res = {"title": "", "authors": "", "content": ""}
    
    # Always extract metadata first with PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        if doc.metadata:
             if doc.metadata.get("title"):
                 res["title"] = doc.metadata.get("title")
             if doc.metadata.get("author"):
                 res["authors"] = doc.metadata.get("author")
    except Exception:
        pass

    # We are completely bypassing Docling to prevent C++ memory crashes (std::bad_alloc)
    # spamming your terminal. PyMuPDF is extremely robust for pure text PDFs anyway!
    
    # Try docling first for robust content
    # try:
    #     from docling.document_converter import DocumentConverter
    #     logger.info("Using docling to convert PDF...")
    #     converter = DocumentConverter()
    #     result = converter.convert(pdf_path)
    #     res["content"] = result.document.export_to_markdown()
    #     
    #     if not res["title"]:
    #         first_line = res["content"].strip().split("\n")[0]
    #         res["title"] = first_line.replace("#", "").strip()
    #         
    #     return res
    # except Exception as e:
    #     logger.warning(f"Docling conversion failed or not installed: {e}. Falling back to PyMuPDF.")

    # Continuing PyMuPDF Fallback for content
    logger.info("Using PyMuPDF to parse PDF content...")
    
    # Extract text with rough headers via size heuristics
    full_text = ""
    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for b in blocks:
            if b['type'] == 0:  # text block
                for l in b["lines"]:
                    for s in l["spans"]:
                        text = s["text"].strip()
                        if not text:
                            continue
                        size = s["size"]
                        if size > 11:
                            if not res["title"]:
                                res["title"] = text
                            full_text += f"\n# {text}\n"
                        else:
                            full_text += f"{text} "
                full_text += "\n"
    
    if not res["title"]:
         res["title"] = "Research Poster"
         
    res["content"] = full_text
    return res
