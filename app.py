import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import OPENAI_API_KEY, PINECONE_API_KEY

# Page config
st.set_page_config(
    page_title="Generative AI Q&A Bot",
    page_icon="🤖",
    layout="wide"
)

# Initialize once (cached so it doesn't reload every time)
@st.cache_resource
def load_rag_system():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("generative-ai-qbank")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model_name="gpt-5.6-luna", temperature=0, openai_api_key=OPENAI_API_KEY)
    return index, embeddings, llm

index, embeddings, llm = load_rag_system()

def ask_question(question):
    query_vector = embeddings.embed_query(question)
    
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    
    context_chunks = [match["metadata"]["text"] for match in results["matches"]]
    context = "\n\n".join(context_chunks)
    
    prompt = f"""Answer the question based only on the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content, context_chunks

# UI
st.title("🤖 Generative AI Question Bank Bot")
st.write("Ask any question from the Generative AI knowledge base")
st.divider()

question = st.text_input("Your question:", placeholder="e.g., What is a transformer model?")

if question:
    with st.spinner("Searching knowledge base..."):
        answer, sources = ask_question(question)
    
    st.subheader("Answer")
    st.write(answer)
    
    with st.expander("View source chunks used"):
        for i, chunk in enumerate(sources, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(chunk)
            st.divider()

with st.sidebar:
    st.header("About")
    st.write("""
    This is a RAG (Retrieval-Augmented Generation) system built with:
    - LangChain
    - OpenAI embeddings & GPT
    - Pinecone vector database
    - Streamlit UI
    
    It answers questions grounded in a Generative AI question bank PDF.
    """)