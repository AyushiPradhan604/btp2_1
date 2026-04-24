import sys
import subprocess

def test():
    try:
        import pymupdf4llm
    except ImportError:
        print("Installing pymupdf4llm...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pymupdf4llm"], check=True)
        import pymupdf4llm

    pdf_path = "0031_Paper_41_paper.pdf"
    print("Testing pymupdf4llm...")
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        print("Success! Extracted chars:", len(md_text))
        print("Sample:\n", md_text[2000:2500])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test()
