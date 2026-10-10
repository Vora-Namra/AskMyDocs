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


# -------------------- Streamlit Configuration --------------------

st.set_page_config(
    page_title="AskMyDocs",
    page_icon="📄",
    layout="wide",
)

# -------------------- API Configuration --------------------

load_dotenv()

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key is missing. Configure GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# -------------------- Read PDF Files --------------------

def get_pdf_text(pdf_docs):
    text_parts = []

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text_parts.append(content)

    return "\n\n".join(text_parts)


# -------------------- Split Text Into Chunks --------------------

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)


# -------------------- Create Vector Store --------------------

def get_vector_store(chunks):
    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding,
    )

    vector_store.save_local("faiss_index")


# -------------------- Extract Readable Response Text --------------------

def extract_answer(content):
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                text_parts.append(item["text"])

        return "\n".join(text_parts).strip()

    return ""


# -------------------- Answer User Questions --------------------

def user_input(user_question):
    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embedding,
        allow_dangerous_deserialization=True,
    )

    docs = new_db.similarity_search(
        user_question,
        k=4,
    )

    if not docs:
        st.info("No relevant information was found in your documents.")
        return

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the provided context.
If the answer is not available in the context, say:
"Answer is not available in the provided context."

Do not invent facts.

Context:
{context}

Question:
{user_question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
    )

    response = model.invoke(prompt)

    # IMPORTANT: Extract and display text inside this function.
    # Ignore signature and other metadata in structured responses.
    answer = extract_answer(response.content)

    st.markdown("### 💬 Reply")

    if answer:
        st.markdown(answer)
    else:
        st.warning("The model returned no readable text.")


# -------------------- Streamlit UI --------------------

def main():
    st.title("📄🤖 AskMyDocs — Chat with Your PDFs")

    st.caption(
        "Upload your documents, process them, and ask questions "
        "based on their content."
    )

    if "vector_store_created" not in st.session_state:
        st.session_state.vector_store_created = False

    with st.sidebar:
        st.header("📁 Upload PDFs")

        pdf_docs = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("📌 Process PDFs", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Reading and indexing documents..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)

                        if not raw_text.strip():
                            st.warning(
                                "No readable text was found. "
                                "The PDFs may be scanned documents "
                                "that require OCR."
                            )
                        else:
                            chunks = get_chunks(raw_text)

                            if not chunks:
                                st.warning(
                                    "No text chunks were generated."
                                )
                            else:
                                get_vector_store(chunks)
                                st.session_state.vector_store_created = True
                                st.success(
                                    "PDFs processed successfully!"
                                )

                    except Exception as e:
                        # Avoid printing API credentials or raw
                        # API error payloads into the user interface.
                        print(
                            "PDF processing failed:",
                            type(e).__name__,
                        )
                        st.error(
                            "PDF processing failed. Check the "
                            "Streamlit Cloud logs for details."
                        )

    user_question = st.text_input(
        "Ask a question about your documents:",
        placeholder="What is the main topic of the document?",
    )

    if user_question.strip():
        if st.session_state.vector_store_created:
            with st.spinner("Searching documents and generating an answer..."):
                try:
                    user_input(user_question.strip())
                except Exception as e:
                    print(
                        "Question answering failed:",
                        type(e).__name__,
                    )
                    st.error(
                        "Unable to answer the question. Check your "
                        "API configuration and Streamlit Cloud logs."
                    )
        else:
            st.info("Upload and process your PDFs before asking questions.")


if __name__ == "__main__":
    main()
