from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os

CHROMA_PATH = "chroma_db"
DOCS_PATH = "docs"

def ingest_documents():
    documents = []

    #go through all PDFs in folder docs/
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCS_PATH, filename)
            print(f"Loading: {filepath}")

            #load PDF
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            documents.extend(pages)

    if not documents:
        print("No PDF files found in docs/ folder")
        return

    #chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50)
        
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")

    #convert each chunk into embedding 
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")

    #save chunks and embeddings to chroma_db
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH)

    print(f"Saved {len(chunks)} chunks to ChromaDB")

if __name__ == "__main__":
    ingest_documents()