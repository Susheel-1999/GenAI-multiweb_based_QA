import os
import streamlit as st

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_core.tools import tool
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
hf_api_key = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Set environment variables
if not hf_api_key :
    raise ValueError("HF_TOKEN is not set. Please add it to your .env file.")
else:
    os.environ['HF_TOKEN'] = hf_api_key

if not groq_api_key :
    raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")

# Define embedding model
model_name = 'sentence-transformers/all-mpnet-base-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Define persistent directory for Chroma
persist_dir = "chroma_data"

# Chat completion LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Function to process URLs
def process_urls(urls:List) -> Chroma:
    """Processes URLs by loading their content into a vector store."""

    vector_store = Chroma(
        collection_name="web-content",
        embedding_function=embedding,
        persist_directory=persist_dir
    )
    
    for url in urls:
        loader = WebBaseLoader(url)
        web_contents = loader.load()
        documents = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=100
        ).split_documents(web_contents)
        vector_store.add_documents(documents=documents)

    return vector_store

# Streamlit app setup
st.set_page_config(page_title="URL chatbot")
st.title("Chat with Webpages")
st.caption(
    "This tool allows you to input a list of URLs and ask queries related to their content. "
    "It processes the web pages using an LLM and provides accurate answers based on the extracted information."
)

# Input for list of URLs
url_input = st.text_area("Enter URLs (comma-separated)", "")
urls = [url.strip() for url in url_input.split(",")] if url_input else []
st.session_state.urls = urls

# Button to process URLs
if st.button("Process URLs"):
    with st.spinner("Processing..."):
        vector_store = process_urls(urls)
        st.session_state.vector_store = vector_store

        # Create retrieval QA chain
        qa = RetrievalQA.from_chain_type(
            return_source_documents=True,
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
            verbose=True
        )

        @tool
        def input_url_info():
            """Provides basic information about the input URLs."""
            url_names = ", ".join(urls)
            url_count = len(urls)
            info = f"{url_names}\nTotal URLs: {url_count}"
            return info

        # Define tools
        tools = [
            Tool(
                name="External Knowledge",
                func=qa.invoke,
                description=(
                    "This tool uses knowledge fetched from the user-provided URLs. "
                    "Use this for specific or pointed questions."
                )
            ),
            Tool(
                name="General Knowledge",
                func=DuckDuckGoSearchRun().invoke,
                description="Use this tool for generic questions."
            ),
            Tool(
                name="Basic URL Info",
                func=input_url_info,
                description="Use this tool to get information about the input URLs."
            )
        ]

        # Initialize agent
        agent = initialize_agent(
            agent="chat-conversational-react-description",
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate",
            memory=conversational_memory
        )
        st.session_state.agent = agent
        st.success("URLs processed successfully!")


else:
    st.warning("session refreshed, loading cache.")
    

# Input for query
query = st.text_input("Enter your query", "")
if st.button("Get Answer"):
    if "agent" in st.session_state:
        with st.spinner("Processing..."):
            output = st.session_state.agent(query)
            if output:
                st.write("Answer:")
                st.write(output["output"])
            else:
                st.error("No answer found.")
    else:
        st.error("Please process the URLs first.")
