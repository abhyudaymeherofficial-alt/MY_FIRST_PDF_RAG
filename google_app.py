
# pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub langchain_text_splitters langchain-openai langchain-community langchain_huggingface sentence-transformers torchvision langchain.memory
#pip install langchain-google-genai
# pip install pycryptodome
# pip install pytesseract pdf2image pillow
# sudo apt install tesseract-ocr
import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import io
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

from PyPDF2 import PdfReader

def get_pdf_text(pdf):
    text = ""

    try:
        # Read PDF bytes once
        pdf_bytes = pdf.read()

        # Create reader from bytes
        reader = PdfReader(io.BytesIO(pdf_bytes))

        # Handle encrypted PDFs
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise Exception(
                    "This PDF is encrypted. Install 'pycryptodome' or provide the correct password."
                )

        # Convert all pages to images (used only if OCR is needed)
        images = convert_from_bytes(pdf_bytes)

        for page_num, page in enumerate(reader.pages):

            # Try normal text extraction first
            page_text = page.extract_text()

            if page_text and page_text.strip():
                text += page_text + "\n"

            else:
                # OCR fallback
                ocr_text = pytesseract.image_to_string(images[page_num])

                if ocr_text.strip():
                    text += ocr_text + "\n"

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),  # FIX: must call .as_retriever()
        memory=memory
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("📄 PDF Chatbot")

    # FIX: Initialize session state at the top, before any conditional logic
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_pdf is not None:
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text(uploaded_pdf)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            # FIX: Store conversation in session_state so it persists across reruns
            st.session_state.conversation = get_conversation_chain(vectorstore)
        st.success("PDF processed! Ask your questions below.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about your PDF...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # FIX: Check if conversation exists and actually invoke it
        if st.session_state.conversation is None:
            answer = "⚠️ Please upload a PDF first."
        else:
            result = st.session_state.conversation.invoke({"question": prompt})
            answer = result["answer"]  # FIX: Extract the answer from the result dict

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# streamlit run google_app.py
if __name__ == "__main__":
    main()
