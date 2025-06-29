import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to read text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# Function to split large text into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save the vector store
def get_vector_store(chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding)
    vector_store.save_local("faiss_index")

# Load the LLM chain with custom prompt
def get_chain():
    prompt = """Answer the question as detailed as possible from the provided context. 
If the answer is not in the context, just say "answer is not available in the provided context". 
Do not make up any answers.

Context:
{context}

Question: {question}

Answer:"""

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt_template = PromptTemplate(template=prompt, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, prompt=prompt_template, chain_type="stuff")
    return chain

# Process user's question and show result
def user_input(user_question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("💬 **Reply:**", response["output_text"])

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.title("📄🤖 Chat with Multiple PDFs using Gemini AI")

    user_question = st.text_input("Ask a question about your documents:")

    if "vector_store_created" not in st.session_state:
        st.session_state.vector_store_created = False

    with st.sidebar:
        st.header("📁 Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])

        if st.button("📌 Process PDFs"):
            if pdf_docs:
                with st.spinner("🔍 Reading and indexing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_chunks(raw_text)
                    get_vector_store(chunks)
                    st.session_state.vector_store_created = True
                    st.success("✅ PDFs processed and indexed!")
            else:
                st.warning("⚠️ Please upload at least one PDF.")

    if user_question:
        if st.session_state.vector_store_created:
            with st.spinner("🤖 Thinking..."):
                user_input(user_question)
        else:
            st.warning("⚠️ Please process the PDFs first.")

if __name__ == "__main__":
    main()
