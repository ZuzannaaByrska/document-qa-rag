from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "chroma_db"

#prompt for LLM
#{context} = chunks of document
#{question} = user question
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
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    #Find the 3 chunks most similar to the question
    #similarity_search compares the question's embedding
    #with the embeddings of all chunks
    results = vectorstore.similarity_search(question, k=3)

    if not results:
        return {
            "answer": "No relevant documents found.",
            "sources": []
        }

    #combine the found fragments into a single context
    context = "\n\n".join([doc.page_content for doc in results])

    #list of sources (the websites from which the answer is taken)
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 0)
        }
        for doc in results
    ]

    #create a prompt with context and a question
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    #connect to Groq LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.0  # niska temperatura = bardziej przewidywalne odpowiedzi
    )

    chain = prompt | llm

    #call the chain with the context and the question
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": response.content,
        "sources": sources
    }