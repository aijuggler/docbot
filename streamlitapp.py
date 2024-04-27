import streamlit as st
from genai import indexing, query_llm
from template import css, bot_template, user_template
import tempfile


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        print(i)
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            print(message.content)
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    st.set_page_config(page_title="Chat with your PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_file = st.file_uploader("Upload your Document", type=["pdf", "doc"])
        if uploaded_file:
            file_contents = uploaded_file.read()
            if file_contents is not None:
                if st.button("Process"):
                    with st.spinner("Processing"):
                        with tempfile.NamedTemporaryFile(delete=False,dir="data_temp") as temp_file:
                            temp_file.write(file_contents)
                            temp_file.seek(0)
                            # get pdf text
                            vectorStores = indexing(pdf=temp_file.name,pdf_name=uploaded_file.name)
                            chain, memory = query_llm(vector_store=vectorStores)
                            # result = chain.invoke(query_input)

                            # create conversation chain
                            st.session_state.conversation = chain


if __name__ == '__main__':
    main()