# Agentic RAG Chatbot

Agentic RAG Chatbot is an AI-powered chatbot designed to provide accurate and relevant information by leveraging various technologies and APIs. This project integrates language models, document processing, and interactive web interfaces to create a robust and efficient chatbot.

## Features

- **Conversational AI**: Engages in natural language conversations with users.
- **Document Retrieval**: Retrieves information from a collection of documents.
- **Search Integration**: Uses DuckDuckGo for web searches.
- **PDF Processing**: Loads and processes PDF documents for information extraction.
- **Embeddings**: Utilizes OpenAI embeddings for semantic understanding.
- **Interactive Interface**: Built with Streamlit for an interactive user experience.

## Technologies Used

- **LangChain**: Framework for building language model applications.
- **Streamlit**: For building an interactive web interface.
- **PyPDFDirectoryLoader**: To load and process PDF documents.
- **OpenAIEmbeddings**: For creating and managing embeddings.
- **RecursiveCharacterTextSplitter**: For splitting documents into manageable chunks.
- **Chroma**: Vector database for efficient document retrieval.
- **WikipediaAPIWrapper**: Utility for interacting with the Wikipedia API.
- **ArxivAPIWrapper**: Utility for interacting with the Arxiv API.
- **DuckDuckGoSearch**: Helps searching information via browser.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/SimranAnand1/Agentic_RAG_Chatbot.git
   cd Agentic_RAG_Chatbot

2. **Create a virtual environment**:
python -m venv agent_env
source agent_env/bin/activate  # On Windows use `agent_env\Scripts\activate

3. **Install dependencies**:
pip install -r requirements.txt

4. **Run commands**:
streamlit run Agent/agentic_rag.py (for document search)
streamlit run crewAI_agent.py  (for coding agent)
streamlit run langchain_agent.py  (for web search tool, arxiv and wikipedia search)
streamlit run pydanticAI_agent.py  (for agentic bot with typesafe valdation and streaming)