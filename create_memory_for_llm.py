from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./vectorStore",
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    
)

prompt = ChatPromptTemplate.from_template("""
Answer the question using the provided context.

Context:
{context}

Question:
{input}
""")

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

def get_response(user_query):

    result = retrieval_chain.invoke(
        {"input": user_query}
    )

    return result["answer"]

def stream_response(query):

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Answer the question using the provided context.

    Context:
    {context}

    Question:
    {query}
    """

    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content

if __name__ == "__main__":
    question = "What is diabetes?"

    answer = get_response(question)

    print(answer)