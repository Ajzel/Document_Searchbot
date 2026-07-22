import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Configure Google AI (if API key is available)
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, backend="google"):
    """Create and save FAISS vector store from text chunks"""
    try:
        if backend == "huggingface":
            # Use HuggingFace (free, local, no API key needed)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.info("Using HuggingFace embeddings (free, local)")
        else:
            # Use Google Gemini embeddings (requires API key)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            st.info("Using Google Gemini embeddings")

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

        # Save backend choice for later use
        with open("backend_choice.txt", "w") as f:
            f.write(backend)

        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def get_conversational_chain(backend="google"):
    """Create the question-answering chain with custom prompt"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
    If the answer is not in the provided context, just say "I am sorry, I do not have that information in the uploaded documents."
    Do not provide wrong answers.

    Context:
    {context}

    Question: {input}

    Answer:
    """

    if backend == "huggingface":
        # For HuggingFace, we could use a local model or still use Google Gemini for generation
        # Since HuggingFace transformers for chat are more complex, we'll still use Gemini for now
        # But you could replace this with a local model if needed
        if google_api_key:
            model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)
        else:
            st.error("Google API key required for text generation. Please set GOOGLE_API_KEY.")
            return None
    else:
        model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = create_stuff_documents_chain(model, prompt)
    return chain

def user_input(user_question):
    """Process user question and return answer from PDF content"""
    try:
        # Load the backend choice
        backend = "google"  # default
        if os.path.exists("backend_choice.txt"):
            with open("backend_choice.txt", "r") as f:
                backend = f.read().strip()

        # Initialize embeddings based on backend
        if backend == "huggingface":
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        # Check if vector store exists
        if not os.path.exists("faiss_index"):
            st.error("Please upload and process PDFs first using the 'Process Documents' button in the sidebar.")
            return

        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain(backend)
        if chain is None:
            return

        response = chain.invoke({"context": docs, "input": user_question})

        st.write("**Answer:**", response)

        # Show source documents (optional)
        with st.expander("📄 Source Context"):
            for i, doc in enumerate(docs):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.write("---")

    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        if "API key" in str(e):
            st.info("💡 Try using HuggingFace backend which doesn't require an API key!")

def main():
    st.set_page_config(
        page_title="AI Document SearchBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 AI Document SearchBot")
    st.markdown("Upload PDF documents and ask questions about their content using AI!")

    # API key status and backend selection
    col1, col2 = st.columns([2, 1])
    with col1:
        if google_api_key:
            st.success("✅ Google API key detected")
        else:
            st.warning("⚠️ No Google API key found - HuggingFace backend will be used")

    with col2:
        if google_api_key:
            backend_choice = st.selectbox(
                "Choose AI Backend:",
                ["google", "huggingface"],
                format_func=lambda x: "Google Gemini" if x == "google" else "HuggingFace (Free)"
            )
        else:
            backend_choice = "huggingface"
            st.info("Using HuggingFace (Free)")

    # Main input area
    user_question = st.text_input(
        "Ask anything about your uploaded PDFs:",
        placeholder="e.g., What are the main topics discussed in the document?"
    )

    if user_question:
        user_input(user_question)

    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("📁 Upload Documents")

        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf']
        )

        if pdf_docs:
            st.info(f"Uploaded {len(pdf_docs)} PDF(s)")

            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)

                    if raw_text.strip():
                        # Create text chunks
                        text_chunks = get_text_chunks(raw_text)
                        st.info(f"Created {len(text_chunks)} text chunks")

                        # Create vector store with selected backend
                        if get_vectorstore(text_chunks, backend=backend_choice):
                            st.success("✅ Documents processed successfully! You can now ask questions.")
                        else:
                            st.error("❌ Failed to process documents")
                    else:
                        st.error("❌ No text could be extracted from the uploaded PDFs")

        # System status
        st.markdown("---")
        st.markdown("### 📊 System Status:")
        if os.path.exists("faiss_index"):
            st.success("✅ Vector database ready")
        else:
            st.warning("⚠️ No documents processed yet")

if __name__ == "__main__":
    main()