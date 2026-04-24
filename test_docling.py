import os
import sys

def test_docling():
    from docling.document_converter import DocumentConverter
    # Test on a small PDF first or the one in folder
    pdf_path = "0031_Paper_41_paper.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} not found.")
        return
        
    try:
        converter = DocumentConverter()
        print("Starting conversion...")
        result = converter.convert(pdf_path)
        print("Conversion successful.")
        md = result.document.export_to_markdown()
        print(f"Extracted {len(md)} chars of markdown.")
    except Exception as e:
        print(f"Docling error: {e}")

if __name__ == "__main__":
    test_docling()
