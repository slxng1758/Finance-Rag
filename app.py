"""
Financial RAG Assistant - Streamlit UI
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import os

# Page configuration
st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_qa_chain():
    """Load the QA chain with caching"""
    
    if not os.path.exists("./chroma_db"):
        st.error("⚠️ Vector database not found! Run `python ingest.py` first.")
        st.stop()
    
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load vector store
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Setup LLM
    llm = OllamaLLM(
        model="llama3.2",
        temperature=0.1
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    
    return qa_chain


def main():
    # Header
    st.title("💰 Financial AI Assistant")
    st.markdown("Ask questions about financial documents")
    
    # Sidebar
    with st.sidebar:
        st.header("📚 About")
        st.markdown("""
        This AI assistant uses **RAG** to answer questions about financial documents.
        
        **How it works:**
        1. Question → embeddings
        2. Retrieve similar chunks
        3. AI generates answer
        4. Sources provided
        """)
        
        st.divider()
        
        st.header("💡 Example Questions")
        st.markdown("""
        - What was the total revenue?
        - Summarize key risk factors
        - What did management say about AI?
        - Compare Q3 vs Q2 performance
        """)
    
    # Load QA chain
    try:
        qa_chain = load_qa_chain()
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure Ollama is running: `ollama serve`")
        st.stop()
    
    # Query input
    st.divider()
    query = st.text_input(
        "Your question:",
        placeholder="What was the revenue of (company name) in Q3?"
    )
    
    if query:
        with st.spinner("🤔 Thinking..."):
            try:
                result = qa_chain({"query": query})
                
                # Display answer
                st.success("✅ Answer:")
                st.write(result['result'])
                
                # Display sources
                with st.expander("📚 View Sources", expanded=False):
                    for i, doc in enumerate(result['source_documents'], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f"📄 File: `{doc.metadata.get('source', 'Unknown').split('/')[-1]}`")
                        st.markdown(f"📖 Page: `{doc.metadata.get('page', 'N/A')}`")
                        st.text_area(
                            f"Content {i}",
                            doc.page_content[:400] + "...",
                            height=150,
                            key=f"source_{i}"
                        )
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error processing question: {e}")


if __name__ == "__main__":
    main()