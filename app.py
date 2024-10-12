import streamlit as st
import re 
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
import os

os.environ["EMB_MODEL"] = 'BAAI/bge-base-en-v1.5'
os.environ["LLM_MODEL"] = 'gpt-3.5-turbo-0125'

# Initialize prompt, embeddings and LLM
prompt = hub.pull("hwchase17/openai-functions-agent")
huggingface_embeddings = HuggingFaceBgeEmbeddings(model_name=os.environ.get("EMB_MODEL"), model_kwargs={'device': 'cpu'})

# Function to sanitize the retriever name with only lowercase a-z
def sanitize_retriever_name(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())  # Remove any characters that are not a-z

# Function to format URLs and create tools
def format_url(urls):
    tools = []
    for index, url in enumerate(urls):
        loader = WebBaseLoader(url)
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

        retriever = vector_store.as_retriever()
        retriever_tool = create_retriever_tool(retriever, retriever_name, f"This is about {retriever_name}")
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

# Input for list of URLs
url_input = st.text_area("Enter URLs (comma-separated)", "")
urls = [url.strip() for url in url_input.split(",")] if url_input else []

# Input for query
query = st.text_input("Enter your query", "")

# Button to process URLs
if st.button("Process URLs"):
    if urls:
        st.session_state.processing = True
        st.session_state.urls = urls  # Update stored URLs
        with st.spinner("Processing URLs..."):
            st.session_state.tools = format_url(urls)  # Create new tools for the new URLs
            st.session_state.agent_executor = llm_with_agents(llm, st.session_state.tools, prompt)  # Create new agent executor
        st.session_state.processing = False

# Display the "Processing" message when necessary
if st.session_state.agent_executor and not st.session_state.processing:
    if urls and query:
        if st.button("Get Answer"):
            with st.spinner("Processing..."):
                output = get_answer(st.session_state.agent_executor, query)
                if output:
                    st.write("Answer:")
                    st.write(output["output"])
                else:
                    st.error("No answer found.")
    else:
        st.info("Please provide both URLs and a query to get an answer.")
else:
    if st.session_state.processing:
        st.info("Processing URLs, please wait...")
    else:
        st.info("Waiting for agent executor to be ready...")