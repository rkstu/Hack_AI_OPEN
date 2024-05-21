import streamlit as st
import replicate
import os
from transformers import AutoTokenizer

# App title
st.set_page_config(page_title="Snowflake Arctic")

def main():
    """Execution starts here."""
    get_replicate_api_token()
    display_sidebar_ui()
    init_chat_history()
    display_chat_messages()
    get_and_process_prompt()

def get_replicate_api_token():
    # os.environ['REPLICATE_API_TOKEN'] = st.secrets['REPLICATE_API_TOKEN']
    os.environ['REPLICATE_API_TOKEN'] = "r8_KID0S5J6F0rMexSDNngNWghN4GQra1E2iTrTA"

def display_sidebar_ui():
    with st.sidebar:
        st.title('Snowflake Arctic')
        st.subheader("Adjust model parameters")
        st.slider('temperature', min_value=0.01, max_value=5.0, value=0.3,
                                step=0.01, key="temperature")
        st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01,
                          key="top_p")

        st.button('Clear chat history', on_click=clear_chat_history)

        st.sidebar.caption('Build your own app powered by Arctic and [enter to win](https://arctic-streamlit-hackathon.devpost.com/) $10k in prizes.')

        st.subheader("About")
        st.caption('Built by [Snowflake](https://snowflake.com/) to demonstrate [Snowflake Arctic](https://www.snowflake.com/blog/arctic-open-and-efficient-foundation-language-models-snowflake). App hosted on [Streamlit Community Cloud](https://streamlit.io/cloud). Model hosted by [Replicate](https://replicate.com/snowflake/snowflake-arctic-instruct).')

        # # # Uncomment to show debug info
        # st.subheader("Debug")
        # st.write(st.session_state)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm Arctic, a new, efficient, intelligent, and truly open language model created by Snowflake AI Research. Ask me anything."}]
    st.session_state.chat_aborted = False

def init_chat_history():
    """Create a st.session_state.messages list to store chat messages"""
    if "messages" not in st.session_state:
        clear_chat_history()

def display_chat_messages():
    # Set assistant icon to Snowflake logo
    icons = {"assistant": "./Snowflake_Logomark_blue.svg", "user": "⛷️"}

    # Display the messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.write(message["content"])

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

def abort_chat(error_message: str):
    """Display an error message requiring the chat to be cleared. 
    Forces a rerun of the app."""
    assert error_message, "Error message must be provided."
    error_message = f":red[{error_message}]"
    if st.session_state.messages[-1]["role"] != "assistant":
        st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        st.session_state.messages[-1]["content"] = error_message
    st.session_state.chat_aborted = True
    st.rerun()

def get_and_process_prompt():
    """Get the user prompt and process it"""
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="./Snowflake_Logomark_blue.svg"):
            response = generate_arctic_response()
            st.write_stream(response)

    if st.session_state.chat_aborted:
        st.button('Reset chat', on_click=clear_chat_history, key="clear_chat_history")
        st.chat_input(disabled=True)
    elif prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

def generate_arctic_response():
    """String generator for the Snowflake Arctic response."""
    prompt = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("user\n" + dict_message["content"] + "")
        else:
            prompt.append("assistant\n" + dict_message["content"] + "")
    
    prompt.append("assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)

    num_tokens = get_num_tokens(prompt_str)
    max_tokens = 1500
    
    if num_tokens >= max_tokens:
        abort_chat(f"Conversation length too long. Please keep it under {max_tokens} tokens.")
    
    st.session_state.messages.append({"role": "assistant", "content": ""})
    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                                  input={"prompt": prompt_str,
                                         "prompt_template": r"{prompt}",
                                         "temperature": st.session_state.temperature,
                                         "top_p": st.session_state.top_p,
                                         }):
        st.session_state.messages[-1]["content"] += str(event)
        yield str(event)

    # Generate an image related to the user prompt
    if st.session_state.messages[-1]["role"] == "user":
        image_output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b", 
            input={"prompt": st.session_state.messages[-1]["content"]}
        )
        st.image(image_output)

if __name__ == "__main__":
    main()
