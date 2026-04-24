import sys
import os

# Add root project path explicitly
sys.path.append(os.path.dirname(__file__))

from utils.pdf_parser import parse_pdf
from utils.llm_client import get_structured_completion
from pydantic import BaseModel

class DummyResponse(BaseModel):
    status: str

def run_tests():
    print("=== Testing LangChain Setup ===")
    try:
        messages = [{"role": "user", "content": "Respond strictly with the status: LangChain OK"}]
        res = get_structured_completion(messages, DummyResponse)
        print(f"[OK] Text LLM Connection: {res.status}")
    except Exception as e:
        print(f"[FAIL] Text LLM Connection FAILED: {e}")
        
    try:
        from docling.document_converter import DocumentConverter
        print("[OK] Docling Library: Installed correctly")
    except ImportError:
        print("[FAIL] Docling Library: NOT INSTALLED")
        
    print("\nEnd of automated system tests.")
    print("To fully test the pipeline, provide a sample PDF and run main.py:")
    print("python main.py sample.pdf")
    
if __name__ == "__main__":
    run_tests()
