import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
import os

# Initialize prompt, embeddings and LLM
prompt = hub.pull("hwchase17/openai-functions-agent")
huggingface_embeddings = HuggingFaceBgeEmbeddings(model_name=os.environ.get("EMB_MODEL"), model_kwargs={'device': 'cpu'})

# Function to format URLs and create tools
def format_url(urls):
    tools = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20).split_documents(docs)
        vector_store = Chroma(collection_name=f"retriever{urls.index(url)+1}", embedding_function=huggingface_embeddings)
        vector_store.add_documents(documents=documents)
        retriever = vector_store.as_retriever()
        retriever_tool = create_retriever_tool(retriever, f"retriever_{urls.index(url)+1}", f"This is about retriever_{urls.index(url)+1}")
        tools.append(retriever_tool)
    return tools

# Function to create agent executor
def llm_with_agents(llm, tools, prompt):
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# Function to get answer from agent executor
def get_answer(agent_executor, query):
    return agent_executor.invoke({"input": query})

# Streamlit app
st.set_page_config(page_title="url query")
st.title("Question Answering Based on URLs")
st.caption("We have designed a tool that allows users to input a list of URLs and ask queries related to the content of those URLs. The tool processes the web pages, extracts relevant information, and returns answers to user queries")
openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

if openai_api_key:
    llm = ChatOpenAI(model=os.environ.get("LLM_MODEL"), temperature=0, openai_api_key=openai_api_key)
    st.write("API key provided and LLM loaded.")
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

# Input for list of URLs
url_input = st.text_area("Enter URLs (comma-separated)", "")
urls = [url.strip() for url in url_input.split(",")] if url_input else []

# Input for query
query = st.text_input("Enter your query", "")

# Check if URLs have changed, and if so, reset the tools and agent executor
if urls and urls != st.session_state.urls:
    st.session_state.urls = urls  # Update stored URLs
    st.session_state.tools = format_url(urls)  # Create new tools for the new URLs
    st.session_state.agent_executor = llm_with_agents(llm, st.session_state.tools, prompt)  # Create new agent executor

# Submit button
if st.button("Get Answer"):
    if urls and query:
        # Use the retained agent executor from session state
        output = get_answer(st.session_state.agent_executor, query)
        st.write("Answer:")
        st.write(output)
    else:
        st.error("Please provide both URLs and a query.")
