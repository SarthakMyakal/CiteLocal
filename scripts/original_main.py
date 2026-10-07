from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
import os
import webbrowser
from pathlib import Path

PDF_PATH = "data/Attention-paper.pdf"

INDEX_PATH = "faiss_index"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "gemma3:4b"


def load_pdf(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    return documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(
        documents= chunks,
        embedding=embeddings
    )
    return vectorstore
    

def save_vectorstore(vectorstore):
    vectorstore.save_local(INDEX_PATH)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        embeddings=embeddings,
        folder_path=INDEX_PATH,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def retrieve_documents(vectorstore, query):
    retrieved_docs = vectorstore.similarity_search(
        query,
        k=3
    )
    return retrieved_docs

def build_prompt(retrieved_docs, question_input):
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    prompt = f"""
    You are a helpful AI assistant.

    Answer ONLY using the provided context.

    If the answer is not present in the context,
    say "I don't know based on the provided document."

    Context:
    {context}

    Question:
    {question_input}

    Answer:
    """
    return prompt

def generate_response(prompt):
    llm = ChatOllama(model=LLM_MODEL)
    response = llm.invoke(prompt)
    return response.content

def ask(question, vectorstore):
    retrieved_docs = retrieve_documents(vectorstore=vectorstore, query=question)
    prompt = build_prompt(retrieved_docs, question)
    answer = generate_response(prompt)
    return answer, retrieved_docs

def open_pdf(page):
    pdf_path = Path(PDF_PATH).resolve()

    pdf_url = pdf_path.as_uri() + f"#page={page}"

    print("PDF PATH :", pdf_path)
    print("PDF URL  :", pdf_url)

    webbrowser.open(pdf_url)  

def main():

    if os.path.exists(INDEX_PATH):
        vectorstore = load_vectorstore()
        print("Loaded existing FAISS index.")

    else:
        documents = load_pdf(PDF_PATH)
        chunks = create_chunks(documents)
        vectorstore = create_vectorstore(chunks)
        save_vectorstore(vectorstore)
        print("Created and saved new FAISS index.")
    
    while True:
        question = input("\nEnter your query (type 'exit' to esc): ")

        if question.lower() == "exit":
            break
            
        answer, retrieved_docs = ask(question, vectorstore)

        print("\nAnswer:\n")
        print(answer)
        if "I don't know based on the provided document." not in answer:
            print("\n Sources:")
            pages = set()
            for doc in retrieved_docs:
                pages.add(doc.metadata["page"]+1)
            pages = sorted(pages)
            for page in pages:
                print(f"Page: {page}")
            open_pdf(pages[0])

if __name__ == "__main__":
    main()