import streamlit as st
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq model and agent
def create_agent(model_name):
    model = GroqModel(model_name, api_key=api_key)
    return Agent(model)

# Function to ensure the event loop is available
def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:  # No event loop in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def main():
    st.title("AI Chatbot with Pydantic-AI and Groq")

    # Sidebar for settings and chat history
    st.sidebar.title("Settings")
    
    # Model selection in the sidebar
    model_name = st.sidebar.selectbox(
        "Select a model:",
        options=["llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=2,  # Default is gemma2-9b-it
    )
    st.sidebar.write(f"**Current Model:** {model_name}")

    # Initialize agent
    agent = create_agent(model_name)

    st.sidebar.title("Chat History")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history in the sidebar
    if st.session_state.chat_history:
        for idx, chat in enumerate(st.session_state.chat_history):
            st.sidebar.write(f"**{idx + 1}. You:** {chat['human']}")
            st.sidebar.write(f"**AI:** {chat['AI']}")

    # User input
    user_input = st.text_input("Your message:")

    # Chatbot interaction
    if user_input:
        with st.spinner("Thinking..."):
            try:
                # Pass previous messages to agent
                if st.session_state.chat_history:
                    last_messages = st.session_state.chat_history[-1]["message_history"]
                else:
                    last_messages = None

                # Create or get an event loop
                loop = get_or_create_event_loop()

                # Run the agent
                result = loop.run_until_complete(
                    agent.run(
                        user_input,
                        message_history=last_messages,
                    )
                )

                # Update session state with the new message
                st.session_state.chat_history.append(
                    {
                        "human": user_input,
                        "AI": result.data,
                        "message_history": result.new_messages(),
                    }
                )

                # Display only the current AI response
                st.write("### AI Response")
                st.write(result.data)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
