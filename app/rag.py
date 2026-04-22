from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "chroma_db"

# prompt który wysyłamy do LLM
# {context} = fragmenty z dokumentów które znaleźliśmy
# {question} = pytanie użytkownika
PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

def answer_question(question: str) -> dict:
    # wczytaj bazę wektorową z dysku
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # znajdź 3 najbardziej podobne chunki do pytania
    # similarity_search porównuje embedding pytania
    # z embeddingami wszystkich chunków
    results = vectorstore.similarity_search(question, k=3)

    if not results:
        return {
            "answer": "No relevant documents found.",
            "sources": []
        }

    # połącz znalezione fragmenty w jeden kontekst
    context = "\n\n".join([doc.page_content for doc in results])

    # lista źródeł (z których stron pochodzi odpowiedź)
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 0)
        }
        for doc in results
    ]

    # stwórz prompt z kontekstem i pytaniem
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # połącz z Groq LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.0  # niska temperatura = bardziej przewidywalne odpowiedzi
    )

    # chain = prompt → llm
    chain = prompt | llm

    # wywołaj chain z kontekstem i pytaniem
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": response.content,
        "sources": sources
    }