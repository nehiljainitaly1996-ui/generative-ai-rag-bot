from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_path = "C:/Users/Nehil/Desktop/GEN-AI/GenAI_Master_Notes_Prompt_RAG_LangChain.pdf"

print(f"📂 Loading PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

print(f"✂️ Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print(f"✅ Split into {len(chunks)} chunks")
print(f"\n📦 Sample chunk (first 300 chars):")
print("="*50)
print(chunks[0].page_content[:300])
print("="*50)