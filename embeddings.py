from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

pdf_path = "C:/Users/Nehil/Desktop/GEN-AI/GenAI_Master_Notes_Prompt_RAG_LangChain.pdf"

print(f"📂 Loading PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"✂️ Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

print(f"\n🧠 Creating test embedding (uses tiny bit of API credit)...")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

test_query = "What is Generative AI?"
sample_embedding = embeddings.embed_query(test_query)

print(f"✅ Embedding created!")
print(f"📊 Dimensions: {len(sample_embedding)}")
print(f"📍 First 5 values: {sample_embedding[:5]}")