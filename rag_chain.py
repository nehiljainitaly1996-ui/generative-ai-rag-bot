from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import OPENAI_API_KEY, PINECONE_API_KEY

# Step 1: Connect to Pinecone
print("🔌 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("generative-ai-qbank")

# Step 2: Setup embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

def ask_question(question):
    # Step 3: Convert question to embedding
    query_vector = embeddings.embed_query(question)
    
    # Step 4: Search Pinecone for top 3 similar chunks
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    
    # Step 5: Extract text from retrieved chunks
    context_chunks = [match["metadata"]["text"] for match in results["matches"]]
    context = "\n\n".join(context_chunks)
    
    # Step 6: Build prompt with context
    prompt = f"""Answer the question based only on the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    
    # Step 7: Ask LLM
    response = llm.invoke(prompt)
    return response.content

# Test it
print("\n✅ RAG System Ready!\n")
print("="*60)

test_questions = [
    "What is Generative AI?",
    "What are transformer models?",
]

for q in test_questions:
    print(f"\n❓ Question: {q}")
    print("-"*60)
    answer = ask_question(q)
    print(f"✅ Answer:\n{answer}")