import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from chat import Chat


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"RAG with Doc"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']


st.title("title bar : Chat with docs, powered by :) 💬🦙")
st.info("Info bar", icon="📃")
         

def change_folder(folder):
    with st.spinner("Thinking..."):
        chroma_directory=f"/Users/user/Desktop/rag_with_documents/{folder}"
        print("Folder name:",folder)
        db = Chroma(collection_name=folder,
                    persist_directory=chroma_directory, 
                    embedding_function=AzureOpenAIEmbeddings(
                                        deployment = "ada002",
                                        model="text-embedding-ada-002",
                                        azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
                                        openai_api_version = "2023-07-01-preview"
                                        )
                    )
        index = Chat(db)
        print(index)
        st.session_state.chat_engine = index

with st.sidebar:
    st.subheader('Document Chatbot!')
    with st.form("change_folder",clear_on_submit=False,border=False):
        folder = st.text_input('Folder name',placeholder="Enter the folder name to chat with.....")

        change_folder_button = st.form_submit_button("Change Folder")
        if change_folder_button:
            change_folder(folder)

    st.divider()
    with st.form("my_form",clear_on_submit=True,border=True):
        # Create a directory if it doesn't exist
        save_dir = "uploaded_files"
        os.makedirs(save_dir, exist_ok=True)
        uploaded_files = st.file_uploader(
            "Upload your PDF documents here",
            type=['pdf'],  #Only Allowing PDF for now
            accept_multiple_files=True
        )
        folder_path = st.text_input(
            label=":rainbow[Folder Name]", 
            value="", 
            on_change=None, 
            placeholder="Insert your folder name here", 
            label_visibility="visible"
        )

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")

        if submitted:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Check if files are uploaded
                    if uploaded_files is not None:
                        chroma_directory=f"/Users/user/Desktop/rag_with_documents/{folder_path}/"
                        db = Chroma(collection_name=folder_path,
                                    persist_directory=chroma_directory, 
                                    embedding_function=AzureOpenAIEmbeddings(
                                                        deployment = "ada002",
                                                        model="text-embedding-ada-002",
                                                        azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
                                                        openai_api_version = "2023-07-01-preview"
                                                        )
                                    )
                        for uploaded_file in uploaded_files:
                            bytes_data = uploaded_file.read()
                            file_name = uploaded_file.name
                            file_path = os.path.join(save_dir, file_name)
                            
                            # Write the contents of the file to a new file
                            with open(file_path, "wb") as f:
                                f.write(bytes_data)
                            
                            loader = PyPDFLoader(file_path)
                            documents = loader.load()

                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000, chunk_overlap=200, add_start_index=True
                            )
                            all_splits = text_splitter.split_documents(documents)

                            db.add_documents(documents=all_splits)
                    else:
                        print("FAILED")

    st.divider()

    cols = st.columns(1)
    if cols[0].button('Refresh'):
        print("session state before clearing",st.session_state)
        st.session_state.clear()
        print("session state after clearing",st.session_state)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": f"Ask me a question about {folder}!"}
    ]

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    default_directory=f"/Users/user/Desktop/rag_with_documents/default"
    db = Chroma(collection_name="default",persist_directory=default_directory, 
                embedding_function=AzureOpenAIEmbeddings(
                                    deployment = "ada002",
                                    model="text-embedding-ada-002",
                                    azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
                                    openai_api_version = "2023-07-01-preview"
                                    )
                )
    index = Chat(db)
    st.session_state.chat_engine = index

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history



