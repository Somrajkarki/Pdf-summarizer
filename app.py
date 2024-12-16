import os
import shutil
import hashlib
import json
import time
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Create directories for file uploads and Chroma vector store if they don't exist
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('data'):
    os.mkdir('data')

# Initialize Streamlit session states
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="llama2",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("MEDITWIN: HEALTHCARE ASSISTANT")

# Function to clear Chroma vectorstore if embedding dimension mismatch occurs
def clear_chroma_store(directory="data"):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        os.mkdir(directory)

# Check if a file hash has changed to avoid reprocessing
def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as file:
        hasher.update(file.read())
    return hasher.hexdigest()

def load_and_analyze_pdfs(directory="files", hash_file="file_hashes.json"):
    documents = []
    # Load or create file hash tracker
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            processed_hashes = json.load(f)
    else:
        processed_hashes = {}

    new_files = False
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(directory, file)
            current_hash = get_file_hash(pdf_path)
            if file not in processed_hashes or processed_hashes[file] != current_hash:
                new_files = True
                loader = PyPDFLoader(pdf_path)
                pdf_data = loader.load()
                documents.extend(pdf_data)
                # Update the hash file to mark the PDF as processed
                processed_hashes[file] = current_hash

    # Save updated hashes
    with open(hash_file, 'w') as f:
        json.dump(processed_hashes, f)
    
    return documents, new_files

# Function to load existing vectorstore or create a new one if required
def load_or_create_vectorstore(documents):
    if os.path.exists("data"):
        # Try to load an existing vectorstore
        try:
            vectorstore = Chroma(
                persist_directory="data",
                embedding_function=OllamaEmbeddings(model="llama2")
            )
            return vectorstore
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}. Recreating...")
            clear_chroma_store()

    # Create a new vectorstore if not available or needs to be rebuilt
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
        )
        all_splits = text_splitter.split_documents(documents)

        try:
            # Create new vectorstore with correct dimensionality
            vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="llama2"),
                persist_directory="data"
            )
            vectorstore.persist()
            return vectorstore
        except Exception as e:
            if "InvalidDimensionException" in str(e):
                # Clear vectorstore in case of dimension mismatch
                clear_chroma_store()
                st.error(f"Dimension mismatch detected: {str(e)}. Vectorstore cleared. Please try reloading.")
            else:
                raise e
    else:
        st.error("No documents available to create vectorstore.")

    return None

# Add file upload option
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Save uploaded files to the 'files' directory
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("files", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} files successfully.")

# Load all PDFs and create a combined vector store if not already created
def check_and_update_vectorstore():
    # Load and analyze PDFs
    documents, new_files = load_and_analyze_pdfs()
    
    # If new files are detected or no vectorstore exists, rebuild or load the vectorstore
    if new_files or st.session_state.vectorstore is None:
        with st.spinner("Loading and processing all PDFs..."):
            st.session_state.vectorstore = load_or_create_vectorstore(documents)
        return True

    # If no new files but vectorstore exists, continue using it
    elif not new_files and st.session_state.vectorstore is not None:
        return False

    # If no documents and no vectorstore, raise an error
    if not documents and not st.session_state.vectorstore:
        st.error("No PDFs found in the 'files' directory. Please upload some PDFs to start.")
    return False

# Check and update vectorstore if necessary
vectorstore_updated = check_and_update_vectorstore()

# Chatbot initialization
if st.session_state.vectorstore is not None:
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    if 'qa_chain' not in st.session_state or vectorstore_updated:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    # User input
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)
else:
    st.write("Please upload some PDFs into the 'files' directory to start.")
