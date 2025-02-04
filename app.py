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
<<<<<<< HEAD
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
=======
        docs = loader.load()
        retriever_name = sanitize_retriever_name(f"retriever_{index+1}")
        documents = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100).split_documents(docs)

        # Create or use existing vector store
        if 'vector_stores' not in st.session_state:
            st.session_state.vector_stores = {}

        if retriever_name not in st.session_state.vector_stores:
            vector_store = Chroma(
                collection_name=retriever_name, 
                embedding_function=huggingface_embeddings, 
                persist_directory="./chroma_data")
            vector_store.add_documents(documents=documents)
            st.session_state.vector_stores[retriever_name] = vector_store
        else:
            vector_store = st.session_state.vector_stores[retriever_name]
        
        retriever_description = docs[0].page_content.replace("\n"," ")[:1000] # max length is 1024
        retriever = vector_store.as_retriever()
        retriever_tool = create_retriever_tool(retriever, retriever_name, f"This is about {retriever_description}")
        tools.append(retriever_tool)
    return tools

# Function to create agent executor
def llm_with_agents(llm, tools, prompt):
    if llm is None:
        st.error("LLM is not initialized. Please provide a valid OpenAI API key.")
        return None
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# Function to get answer from agent executor
def get_answer(agent_executor, query):
    if agent_executor is None:
        st.error("Agent executor is not initialized.")
        return None
    return agent_executor.invoke({"input": query})

# Streamlit app
st.set_page_config(page_title="URL Query")
st.title("Question Answering Based on URLs")
st.caption("Input a list of URLs and ask queries related to their content. The tool processes the web pages and provides answers.")

# OpenAI API key input
openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

if openai_api_key:
    try:
        llm = ChatOpenAI(model=os.environ.get("LLM_MODEL"), temperature=0, openai_api_key=openai_api_key.strip())
        llm.invoke("Hello")  # Test the API key
        st.write("API key provided and LLM loaded.")
    except Exception:
        llm = None
        st.error("API key is invalid.")
else:
    st.error("Please enter your OpenAI API key.")
    llm = None

# Initialize session state for URLs, tools, and agent executor if not already set
if 'urls' not in st.session_state:
    st.session_state.urls = []
if 'tools' not in st.session_state:
    st.session_state.tools = None
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
>>>>>>> abec57f336124ba0f5d6f524be5e1cdc44023e69

# Input for list of URLs
url_input = st.text_area("Enter URLs (comma-separated)", "")
urls = [url.strip() for url in url_input.split(",")] if url_input else []
st.session_state.urls = urls

# Button to process URLs
if st.button("Process URLs") and st.session_state.urls:
    with st.spinner("Processing..."):
        vector_store = process_urls(urls)

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
else:
    st.error("Please enter the URLs.")

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
<<<<<<< HEAD
        st.error("Please process the URLs first.")
=======
        st.info("Waiting for agent executor to be ready...")
>>>>>>> abec57f336124ba0f5d6f524be5e1cdc44023e69
