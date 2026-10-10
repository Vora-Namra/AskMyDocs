import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
# Load environment variables

load_dotenv()

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please configure your GOOGLE_API_KEY.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


# Create and save the vector store


def get_vector_store(chunks):
    try:
        embedding = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )

        vector_store = FAISS.from_texts(chunks, embedding)
        vector_store.save_local("faiss_index")

    except Exception as e:
        print("Embedding error:", repr(e))
        st.error("Failed to create embeddings.")
        st.code(f"{type(e).__name__}: {e}")
        raise

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
    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question, k=4)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
Answer the question using only the provided context.
If the answer is unavailable in the context, say:
"Answer is not available in the provided context."

Do not invent facts.

Context:
{context}

Question: {user_question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    response = model.invoke(prompt)

    st.write("💬 **Reply:**", response.content)

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
