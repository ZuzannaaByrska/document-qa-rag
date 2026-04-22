from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os

# ścieżka gdzie będzie zapisana baza wektorowa
CHROMA_PATH = "chroma_db"
DOCS_PATH = "docs"

def ingest_documents():
    documents = []

    # przejdź przez wszystkie PDF w folderze docs/
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCS_PATH, filename)
            print(f"Loading: {filepath}")

            # wczytaj PDF strona po stronie
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            documents.extend(pages)

    if not documents:
        print("No PDF files found in docs/ folder")
        return

    # podziel dokumenty na małe kawałki (chunki)
    # chunk_size=500 → każdy kawałek ma max 500 znaków
    # chunk_overlap=50 → kawałki nakładają się o 50 znaków
    # żeby nie urwać zdania w połowie
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")

    # zamień każdy chunk na wektor liczb (embedding)
    # all-MiniLM-L6-v2 to darmowy model który działa lokalnie
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # zapisz chunki i ich embeddingi do ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Saved {len(chunks)} chunks to ChromaDB")

if __name__ == "__main__":
    ingest_documents()