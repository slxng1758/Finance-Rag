"""
Document Ingestion Pipeline
Loads PDFs, chunks them, creates embeddings, and stores in vector DB
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from pathlib import Path

def ingest_documents(data_dir="./data", persist_dir="./chroma_db"):
    """
    Ingest all PDFs from data directory into vector database
    """
    
    print("🚀 Starting document ingestion...")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"❌ Created {data_dir} directory - please add PDF files and run again")
        return
    
    # Find all PDF files
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {data_dir}")
        print(f"   Please add PDF files to the {data_dir} folder")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    # Load all documents
    all_docs = []
    for pdf_file in pdf_files:
        print(f"📖 Loading: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            all_docs.extend(docs)
            print(f"   ✅ Loaded {len(docs)} pages")
        except Exception as e:
            print(f"   ❌ Error loading {pdf_file.name}: {e}")
    
    if not all_docs:
        print("❌ No documents loaded successfully")
        return
    
    print(f"\n📚 Total pages loaded: {len(all_docs)}")
    
    # Split documents into chunks
    print("\n✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"   ✅ Created {len(chunks)} chunks")
    
    # Create embeddings
    print("\n🧠 Creating embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create vector store
    print("\n💾 Storing in vector database...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"\n✅ SUCCESS! Indexed {len(chunks)} chunks from {len(pdf_files)} documents")
    print(f"📁 Vector database saved to: {persist_dir}")
    print("\n🎉 Ready to run: streamlit run app.py")


if __name__ == "__main__":
    ingest_documents()