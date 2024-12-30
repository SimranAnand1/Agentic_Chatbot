import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

#Loading the groq api key
groq_api_key = os.getenv('GROQ_API_KEY')


#Defining tools

# Tool 1: Wikipedia Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max=10000)
wiki = WikipediaQueryRun(api_wrapper = wiki_wrapper)


#Tool 2: Arxiv tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=10000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)


tools = [wiki, arxiv]


#Streamlit setup
st.title("Chatbot using Langchain Agent")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")


#Prompt
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate and a detailed response based on the question
<context>
{context}
<context>
Questions:{input}
{agent_scratchpad}
"""
)

# Agent Setup
from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Agent Executer
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)


query = st.text_input("Input your query here")

if (st.button("Get Answer") or query):
    start_overall = time.time()
    start_llm = time.process_time()
    #response = agent_executor.invoke({"input": query})
    try:
        response = agent_executor.invoke({
            "input": query,
            "context": "",
            "agent_scratchpad": ""
        })
        response_time_overall = time.time() - start_overall
        response_time_llm = time.process_time() - start_llm
        st.write(response['output'])
        #st.write(f"Response time: {response_time} seconds")
        st.markdown(f"<p style='color:blue;'>Overall Response Time: {response_time_overall} seconds</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:blue;'>LLM Response Time: {response_time_llm} seconds</p>", unsafe_allow_html=True)

    except Exception as e:
        #st.markdown(f"<p style='color:red;'>Please enter a valid query!</p>", unsafe_allow_html=True)
        st.write(f"An error occurred: {e}")
        
else:
    st.write("Please enter a query")
