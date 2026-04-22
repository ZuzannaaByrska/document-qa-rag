# Document Q&A RAG System

Ask questions about your PDF documents in natural language.
Built with LangChain, ChromaDB, FastAPI and Docker.

## Architecture
PDF -> chunking -> embeddings -> ChromaDB

question -> FastAPI -> similarity search -> Groq LLM -> answer + sources

## Tech Stack
- LangChain - RAG pipeline orchestration
- ChromaDB - vector database for semantic search
- SentenceTransformers - text embeddings (all-MiniLM-L6-v2)
- Groq LLM - answer generation (llama-3.3-70b-versatile)
- FastAPI - REST API endpoint
- Docker - containerization

## API Endpoints

| Method | Endpoint | Description |
| GET    | /        | Health check |
| GET    | /health  | Service status |
| POST   | /ask     | Ask a question |

## How to Run

### 1. Clone and setup
```bash
git clone https://github.com/ZuzannaaByrska/document-qa-rag.git
cd document-qa-rag
```

### 2. Add your PDF
```bash
mkdir docs
cp your_document.pdf docs/
```

### 3. Create .env file
GROQ_API_KEY=your_groq_api_key_here

### 4. Run with Docker
```bash
docker build -t document-qa-rag .
docker run -p 8000:8000 --env-file .env document-qa-rag
```

### 5. Ingest documents
```bash
python -m app.ingest
```

### 6. Ask questions
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}'
```

Or open **http://localhost:8000/docs** for interactive Swagger UI.

## Example Response
```json
{
  "answer": "Based on the document, ...",
  "sources": [
    {"source": "docs/document.pdf", "page": 3},
    {"source": "docs/document.pdf", "page": 7}
  ]
}
```

## Limitations
- Works best with text-based PDFs (not scanned images)
- Answer quality depends on chunk size and document structure
- Groq API key required for LLM generation
