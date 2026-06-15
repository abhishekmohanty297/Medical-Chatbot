import streamlit as st


from create_memory_for_llm import get_response, stream_response

def main():

    st.title("🏥 Medical Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display old messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask your medical question..."
    )

    if prompt:

        # User Message
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append(
            {
                "role":"user",
                "content":prompt
            }
        )

        # Bot Response
        with st.chat_message("assistant"):
            with st.spinner("Searching medical knowledge base..."):
                response = st.write_stream(
            stream_response(prompt)
            )
        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":response
            }
        )
        

if __name__ == "__main__":
    main()
