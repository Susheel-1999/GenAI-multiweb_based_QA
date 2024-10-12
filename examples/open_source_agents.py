from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.llms import HuggingFaceHub
from langchain.agents import AgentExecutor, create_react_agent
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'hf_'
os.environ["MAX_LENGTH"] = '1000'
os.environ["LLM_MODEL"] = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
os.environ["EMB_MODEL"] = 'BAAI/bge-base-en-v1.5'

huggingface_embeddings = HuggingFaceBgeEmbeddings(model_name = os.environ.get("EMB_MODEL"), model_kwargs={'device': 'cpu'})
prompt = hub.pull("hwchase17/react")
llm = HuggingFaceHub(repo_id=os.environ.get("LLM_MODEL"), model_kwargs={"temperature": 0.5, "max_length": int(os.environ.get("MAX_LENGTH"))})

url = ["https://en.wikipedia.org/wiki/Tom_and_Jerry", "https://en.wikipedia.org/wiki/Power_Rangers"]

loader=WebBaseLoader(url[0])
docs=loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=20).split_documents(docs[:1000])
vector_store = Chroma(collection_name="retriever_1", embedding_function=huggingface_embeddings)
print(len(documents))
vector_store.add_documents(documents=documents)
retriever_1 = vector_store.as_retriever()
retriever_tool1 = create_retriever_tool(retriever_1,"tom and jerry",
                      "Tom and Jerry is an American animated media franchise and series of comedy short films created in 1940 by William Hanna and Joseph Barbera. Best known for its 161 theatrical short films by Metro-Goldwyn-Mayer, the series centers on the rivalry between the titular characters of a cat named Tom and a mouse named Jerry. Many shorts also feature several recurring characters.")

loader=WebBaseLoader(url[1])
docs=loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=390,chunk_overlap=20).split_documents(docs[:1000])
vector_store = Chroma(collection_name="retriever_2", embedding_function=huggingface_embeddings)
print(len(documents))
vector_store.add_documents(documents=documents)
retriever_2 = vector_store.as_retriever()
retriever_tool2 = create_retriever_tool(retriever_2,"power ranges",
                      "Power Rangers is an entertainment and merchandising franchise created by Haim Saban, Shuki Levy and Shotaro Ishinomori and built around a live-action superhero television series, based on Japanese tokusatsu franchise Super Sentai and currently owned by American toy and entertainment company Hasbro through a dedicated subsidiary, SCG Power Rangers LLC. It was first produced in 1993 by Saban Entertainment (later BVS Entertainment), which Saban sold to the Walt Disney Company and then brought back under his now-defunct successor company Saban Brands within his current company, Saban Capital Group, the Power Rangers television series takes much of its footage from the Super Sentai television series produced by Toei Company.[1] The first Power Rangers entry, Mighty Morphin Power Rangers, debuted on August 28, 1993, and helped launch the Fox Kids programming block of the 1990s, during which it catapulted into popular culture along with a line of action figures and other toys by Bandai.[2] By 2001, the media franchise had generated over $6 billion in toy sales.[3]. Despite initial criticism that its action")

tools = [retriever_tool1, retriever_tool2]

prompt_template = hub.pull("hwchase17/react-chat")

agent = create_react_agent(llm, tools, prompt = prompt_template)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

agent_executor.invoke({"input": "who is tom and jerry",
        "chat_history": []})

# Output:
# {'input': 'who is tom and jerry',
# 'chat_history': [],
# 'output': 'Agent stopped due to iteration limit or time limit.'}