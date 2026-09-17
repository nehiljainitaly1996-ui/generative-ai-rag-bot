from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY

# Step 1: Load PDF
pdf_path = "C:/Users/Nehil/Desktop/GEN-AI/GenAI_Master_Notes_Prompt_RAG_LangChain.pdf"
print(f"📂 Loading PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

# Step 2: Split into chunks
print(f"✂️ Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# Step 3: Connect to Pinecone
print(f"\n🔌 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("generative-ai-qbank")
print(f"✅ Connected to index!")

# Step 4: Create embeddings
print(f"\n🧠 Creating embeddings for all chunks (this takes 1-2 mins)...")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Step 5: Upload in batches
print(f"\n🚀 Uploading to Pinecone...")

batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    
    texts = [chunk.page_content for chunk in batch]
    vectors = embeddings.embed_documents(texts)
    
    to_upsert = []
    for j, (chunk, vector) in enumerate(zip(batch, vectors)):
        to_upsert.append({
            "id": f"chunk-{i+j}",
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "page": chunk.metadata.get("page", 0)
            }
        })
    
    index.upsert(vectors=to_upsert)
    print(f"  Uploaded batch {i//batch_size + 1} ({len(to_upsert)} chunks)")

print(f"\n✅ Successfully uploaded {len(chunks)} chunks to Pinecone!")