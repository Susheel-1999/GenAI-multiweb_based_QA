# GenAI - Multiweb based Question Answering
This is a system designed to comprehensively respond to questions posed about multiple URLs, regardless of where the questions originate. 

# Steps to run Streamlit app:
1. Create an Groq api key and set up an environment variable, visit: https://console.groq.com/keys <br>
   ![groq_api](https://github.com/user-attachments/assets/782090e9-0f7c-4f8a-81f1-3e9c4d1979d8)


2. Create an Hugging face access token and set up an environment variable, visit: https://console.groq.com/keys <br>
   ![hf token](https://github.com/user-attachments/assets/ff62d94a-3579-492f-9102-7e4f877e965e)

3. Create a new environment: <br>
```conda create -p genai python==3.9 -y```

4. Activate the environment: <br>
```conda activate genai```

5. Install the requirements: <br>
```pip install -r requirements.txt```

6. Run the Streamlit application: <br>
```streamlit run app.py```

# Workflow:
1. Create a grop API key and hugging-face access token as mentioned above.
2. Input N URLs (comma-separated) into the URL field.
3. Click "Process URLs", it will scrape the web content, split it into chunks, and index it into ChromaDB.
4. Enter your question and click "Get Answer" to retrieve the response, which will be sourced from any relevant content found within the provided URLs.

**Quick start:** https://huggingface.co/spaces/susheel-1999/urlQA

# About Techniques:
Langchain is a framework for developing applications powered by language models. It enables applications that are context-aware and reason.

1. **Web Loader:** <br>
     Web Loader is a LangChain package used for extracting the content of a given URL. It processes web pages and returns their text, making them ready for further analysis.
   <br>`from langchain_community.document_loaders import WebBaseLoader` <br>
2. **Chroma DB:** <br>
     ChromaDB offers real-time updates, allowing you to add or remove vectors dynamically without needing to rebuild the index, unlike FAISS. It is optimized for text embeddings and semantic search, making it easier to implement in NLP tasks. FAISS, on the other hand, is more general-purpose and requires additional setup for similar tasks. Additionally, ChromaDB is cloud-native and scales efficiently across distributed environments, whereas FAISS requires more customization for similar scalability.  
3. **Agents:** <br>
    The core idea of agents is to use a language model to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code). In agents, a language model is used as a reasoning engine to determine which actions to take and in which order. (Reasoning and Action) <br>
   ![image](https://github.com/user-attachments/assets/88143d00-dbed-4350-a098-fbc45efd0ef5)
   <br>
   We can think of agents as enabling “tools” for LLMs. Like how a human would use a calculator for maths or perform a Google search for information — agents allow an LLM to do the same thing.
   ![image](https://github.com/user-attachments/assets/2ab3a0f5-d297-43a7-8a0b-beae17f82d25)

   **Different Types of Agents**

     i) **Zero Shot React**
     
     This agent is used to perform “zero-shot” tasks on some input. This means the agent considers one single interaction with the agent — it will have no memory. At each step, there is a Thought that results in a chosen Action and Action Input. If the Action uses a tool, an Observation (output from the tool) is passed back to the agent. It follows the flow:
     
     `Question (from the user) -> Thought -> Action -> Action Input -> Observation`, repeating until reaching the Final Answer.
     
     **Template:**
     ```plaintext
     Answer the following questions as best you can. You have access to the following tools:
     
     Tools1 name: Tools1 description
     Tools2 name: Tools2 description
     
     Use the following format:
     
     Question: the input question you must answer
     Thought: you should always think about what to do
     Action: the action to take, should be one of [Calculator, Stock DB]
     Action Input: the input to the action
     Observation: the result of the action
     ... (this Thought/Action/Action Input/Observation can repeat N times)
     Thought: I now know the final answer
     Final Answer: the final answer to the original input question
     
     Begin!
     
     Question: {input}
     Thought: {agent_scratchpad}
     ```
     
     The `agent_scratchpad` is where we add every thought or action the agent has already performed. All thoughts and actions (within the current agent executor chain) can then be accessed by the next thought-action-observation loop, enabling continuity in agent actions.
     
     **Code:**
     ```python
     from langchain.agents import initialize_agent
     
     zero_shot_agent = initialize_agent(
         agent="zero-shot-react-description", 
         tools=tools, 
         llm=llm,
         verbose=True,
         max_iterations=3
     )
     ```
     
     ---
     
     ii) **Conversational ReAct**
     
     The zero-shot agent works well but lacks conversational memory. This lack of memory can be problematic for chatbot-type use cases that need to remember previous interactions in a conversation. The Conversational ReAct agent is similar to the Zero Shot ReAct agent but includes conversational memory, allowing it to solve questions using a more complex approach.
     
     **Template:**
     ```plaintext
     Assistant is a large language model trained by OpenAI.
     
     Assistant is designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on various topics. It can engage in natural-sounding conversations and provide coherent and relevant responses.
     
     TOOLS:
     ------
     Assistant has access to the following tools:
     
     > Tool1 name: Tool1 description
     > Tool2 name: Tool2 description
     
     To use a tool, please use the following format:
     ---
     Thought: Do I need to use a tool? Yes
     Action: the action to take, should be one of [Tool1 name, Tool2 name]
     Action Input: the input to the action
     Observation: the result of the action
     ---
     
     When responding to the user, or if no tool is needed, use this format:
     ---
     Thought: Do I need to use a tool? No
     AI: [your response here]
     ---
     
     Begin!
     
     Previous conversation history:
     {chat_history}
     
     New input: {input}
     {agent_scratchpad}
     ```
     
     Here, `chat_history` contains all previous interactions added to the prompt.
     
     **Code:**
     ```python
     from langchain.memory import ConversationBufferMemory
     
     memory = ConversationBufferMemory(memory_key="chat_history")
     
     conversational_agent = initialize_agent(
         agent='conversational-react-description', 
         tools=tools, 
         llm=llm,
         verbose=True,
         max_iterations=3,
         memory=memory
     )
     ```
     
     ---
     
     iii) **ReAct Docstore**
     
     The ReAct Docstore agent uses the ReAct methodology, explicitly built for information search and lookup using a LangChain docstore. Unlike the Conversational agent, there is no `chat_history` input, meaning this is another zero-shot agent.
     
     **Code:**
     ```python
     from langchain import Wikipedia
     from langchain.agents.react.base import DocstoreExplorer
     
     docstore = DocstoreExplorer(Wikipedia())
     
     tools = [
         Tool(
             name="Search",
             func=docstore.search,
             description='search Wikipedia'
         ),
         Tool(
             name="Lookup",
             func=docstore.lookup,
             description='lookup a term in Wikipedia'
         )
     ]
     
     docstore_agent = initialize_agent(
         tools, 
         llm, 
         agent="react-docstore", 
         verbose=True,
         max_iterations=3
     )
     ```
     This agent implements two docstore methods:
     - **Search**: Searches for a relevant article.
     - **Lookup**: Finds the relevant chunk of information within the retrieved article.
     
     ---
     
     iv) **Self-Ask With Search**
     
     This agent connects an LLM with a search engine. It performs searches and asks follow-up questions as often as required to get a final answer. The agent performs multiple follow-up questions to refine the final answer.
     
     **Code:**
     ```python
     # Initialize the search-enabled agent
     self_ask_with_search = initialize_agent(
         tools,
         llm,
         agent="self-ask-with-search",
         verbose=True
     )
     ```
     
     **Reference:** [LangChain Agents – Pinecone](https://www.pinecone.io/learn/series/langchain/langchain-agents/)
   <br><br>
   **Notes based on My experience in agents:** <br>
   i) When using LangChain's agent tool with OpenAI models or any good open source LLM models, it operates effectively and yields accurate results. <br>
   ii) Conversely, when utilizing LangChain's agent tool with few open-source models, the execution may run indefinitely and struggle to generate appropriate responses. The code illustrating this issue can be found in the attached repository.
<br>
Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes we can build and deploy powerful data apps. <br>
1. Session State - Session State is a way to share variables between reruns, for each user session.

# UI 
![image](https://github.com/user-attachments/assets/1f2bf726-d25c-4fc1-8a18-6980c839a67f)

# Reference:
Langchain - https://python.langchain.com/docs/get_started/introduction  <br>
OpenAI - https://platform.openai.com/docs/introduction <br>
Streamlit - https://docs.streamlit.io/library/api-reference/session-state <br>
Groq Api - https://console.groq.com/docs/api-reference <br>
Hugging face - https://huggingface.co/
