# load_pdf.py
from langchain.document_loaders import PyPDFLoader
from config import OPENAI_API_KEY

# Change this to your PDF file name
pdf_path = "C:/Users/Nehil/Desktop/GEN-AI/GenAI_Master_Notes_Prompt_RAG_LangChain.pdf"  # तुम्हारी PDF का नाम

try:
    print(f"📂 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"✅ PDF loaded successfully!")
    print(f"📄 Total pages: {len(documents)}")
    print(f"\n📖 First page preview (first 500 chars):")
    print("="*50)
    print(documents[0].page_content[:500])
    print("="*50)
    
except FileNotFoundError:
    print(f"❌ Error: {pdf_path} not found!")
    print("📌 Make sure your PDF is in the same folder as this script")
    print("📌 And update the pdf_path variable with correct name")