import streamlit as st, requests, time

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an Google Gemini API key](https://ai.google.dev/gemini-api/docs/api-key)"
    "[View the source code](https://github.com/ziruiwang1996/ai-agent-mcp)"

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Gemini")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Send query to GeminiChatBot API and handle streaming response
    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            response = requests.post(
                "http://localhost:8000/chat",
                json={"query": prompt},
                stream=True 
            )

            text_accum = ""
            for chunk in response.iter_content(chunk_size=32, decode_unicode=True):
                if not chunk: continue
                text_accum += chunk
                response_placeholder.markdown(text_accum)
                time.sleep(0.01)
            # final render without cursor
            response_placeholder.markdown(text_accum)
            st.session_state.messages.append({"role":"assistant","content":text_accum})

    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
        st.chat_message("assistant").write(f"Error: {str(e)}")