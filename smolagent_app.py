import streamlit as st
import os
from dotenv import load_dotenv
import datasets
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from smolagents import Tool, HfApiModel, CodeAgent, LiteLLMModel
from langchain_community.retrievers import BM25Retriever
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("HF_TOKEN is not set. Please check your .env file.")
    st.stop()

api_key = os.getenv("GROQ_API_KEY")

st.title("Transformers Model Q&A with Agentic RAG")
st.write("Ask a question about Transformer model training below:")

# User selection for data source
data_source = st.radio(
    "Select data source:",
    ("Use preset Hugging Face dataset", "Upload your own PDF file")
)

# Function to load and process documents from the preset dataset
@st.cache_data
def load_and_process_docs_from_dataset():
    knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
    knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))
    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
        for doc in knowledge_base
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_documents(source_docs)

# Function to load and process documents from an uploaded PDF
@st.cache_data
def load_and_process_docs_from_pdf(uploaded_file):
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        source_docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return text_splitter.split_documents(source_docs)
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
        return []

# Load documents based on user selection
docs_processed = []
if data_source == "Use preset Hugging Face dataset":
    docs_processed = load_and_process_docs_from_dataset()
else:
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file is not None:
        docs_processed = load_and_process_docs_from_pdf(uploaded_file)
    else:
        st.warning("Please upload a PDF file to proceed.")

# Proceed if documents are loaded
if docs_processed:
    # Define retriever tool
    class RetrieverTool(Tool):
        name = "retriever"
        description = "Retrieves relevant sections from Transformers documentation."
        inputs = {
            "query": {
                "type": "string",
                "description": "A search query optimized for semantic similarity with target documents."
            }
        }
        output_type = "string"

        def __init__(self, docs, **kwargs):
            super().__init__(**kwargs)
            self.retriever = BM25Retriever.from_documents(docs, k=10)

        def forward(self, query: str) -> str:
            assert isinstance(query, str), "Your search query must be a string"
            docs = self.retriever.invoke(query)
            return "\nRetrieved documents:\n" + "".join(
                [f"\n\n===== Document {i} =====\n" + doc.page_content for i, doc in enumerate(docs)]
            )

    retriever_tool = RetrieverTool(docs_processed)

    # Initialize the agent with LiteLLMModel
    agent = CodeAgent(
        tools=[retriever_tool],
        model=HfApiModel(
            # model_id="anthropic/claude-3-5-sonnet-latest",
            # api_base="https://api-inference.huggingface.co",
            api_key=hf_token
        ),
        max_steps=4,
        verbosity_level=2
    )

    # User input for query
    user_query = st.text_input(
        "Enter your question:",
        "For a transformers model training, which is slower: the forward or the backward pass?"
    )

    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            try:
                agent_output = agent.run(user_query)
                st.success("Answer generated!")
                st.write("### Final Output:")
                st.write(agent_output)
            except Exception as e:
                st.error(f"Error in generating the model output: {e}")
