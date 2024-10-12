# GenAI - Multiweb based Question Answering
This is a system designed to comprehensively respond to questions posed about multiple URLs, regardless of where the questions originate.

# Steps to run Streamlit app:
1. To create an openai api key, visit: https://platform.openai.com/api-keys  
![image](https://github.com/user-attachments/assets/f7f26f76-98f6-46d4-ac78-45e72db70c96)

2. Create a new environment: <br>
```conda create -p genai python==3.9 -y```

3. Activate the environment: <br>
```conda activate genai```

4. Install the requirements: <br>
```pip install -r requirements.txt```

5. Run the Streamlit application: <br>
```streamlit run app.py```

# Workflow:
1. Create an OpenAI API key as mentioned above and and enter it in the provided field.
2. Input N URLs (comma-separated) into the URL field.
3. Click "Process", it will scrape the web content, split it into chunks, and index it into ChromaDB.
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
    The core idea of agents is to use a language model to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code). In agents, a language model is used as a reasoning engine to determine which actions to take and in which order. <br>
   ![image](https://github.com/user-attachments/assets/88143d00-dbed-4350-a098-fbc45efd0ef5)

   **Notes based on My experience in agents:** <br>
   i) When using LangChain's agent tool with OpenAI models, it operates effectively and yields accurate results. <br>
   ii) Conversely, when utilizing LangChain's agent tool with open-source models, the execution may run indefinitely and struggle to generate appropriate responses. The code illustrating this issue can be found in the attached repository.
<br>
Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes we can build and deploy powerful data apps. <br>
1. Session State - Session State is a way to share variables between reruns, for each user session.

# UI 
![image](https://github.com/user-attachments/assets/60c8478f-b8cd-4ecc-8629-1f6bdc52f02d)

# Reference:
Langchain - https://python.langchain.com/docs/get_started/introduction  <br>
OpenAI - https://platform.openai.com/docs/introduction <br>
Streamlit - https://docs.streamlit.io/library/api-reference/session-state
