import streamlit as st
from huggingface_hub import InferenceApi
import os

# Use the API token directly
API_TOKEN = "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"

# Initialize the Inference API for the model (example: bloom model)
api = InferenceApi(repo_id="bigscience/bloom", token=API_TOKEN)

def init_page() -> None:
    st.set_page_config(page_title="Personal Chatbot")
    st.header("Personal Chatbot")
    st.sidebar.title("Options")

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Reply your answer in markdown format."}
        ]

def get_answer(user_input) -> str:
    response = api(inputs=user_input)
    return response.get('generated_text', 'Sorry, no response.')

def main() -> None:
    init_page()
    init_messages()

    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Bot is typing ..."):
            answer = get_answer(user_input)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    messages = st.session_state.get("messages", [])
    for message in messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
        elif message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])

if __name__ == "__main__":
    main()
