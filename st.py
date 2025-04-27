import os
import sys
import tempfile
import streamlit as st
from llm import PDFQuery
from  streamlit_chat import message
from dotenv import load_dotenv,find_dotenv

_=load_dotenv(find_dotenv())

openai_api_key=os.environ['openai_api_key']
base_url=os.environ['base_url']


st.set_page_config(page_title="ChatPDF")

def display_messages():
    st.subheader("Chat")
    for i,(msg,is_user) in enumerate(st.session_state['messages']):
        message(msg,is_user=is_user,key=str(i))
    st.session_state['thinking_spinner']=st.empty()

def display_retrieved_docs():
    if hasattr(st.session_state['pdfquery'], 'retriever_contents') and st.session_state['pdfquery'].retriever_contents:
        with st.expander("查看检索到的文档内容"):
            for content in st.session_state['pdfquery'].retriever_contents:
                st.markdown(content)
        st.session_state['pdfquery'].retriever_contents=[]
def process_input():
    if st.session_state['user_input'] and len(st.session_state['user_input'].strip())>0:
        user_text=st.session_state['user_input'].strip()
        with st.session_state['thinking_spinner'],st.spinner("Thinking"):
            query_text=st.session_state['pdfquery'].ask(user_text)
        st.session_state['messages'].append((user_text,True))
        st.session_state['messages'].append((query_text,False))
        display_retrieved_docs()

def read_and_save_file():
    st.session_state["pdfquery"].forget()  # to reset the knowledge base
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state['file_uploader']:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path=tf.name
        with st.session_state['ingestion_spinner'],st.spinner(f"Ingesting {file.name}"):
            st.session_state['pdfquery'].ingest(file_path)
        os.remove(file_path)

def is_openai_api_key_set():
    return len(st.session_state['OPENAI_API_KEY'])>0



def main():
    if len(st.session_state)==0:
        st.session_state['messages']=[]
        st.session_state["OPENAI_API_KEY"]=openai_api_key
        if is_openai_api_key_set():
            st.session_state['pdfquery']=PDFQuery()
        else:
            st.sesstion_state['pdfquery']=None
    st.header("ChatPDF")

    if st.text_input("OpenAI API Key", value=st.session_state["OPENAI_API_KEY"], key="input_OPENAI_API_KEY", type="password"):
        if (
            len(st.session_state["input_OPENAI_API_KEY"]) > 0
            and st.session_state["input_OPENAI_API_KEY"] != st.session_state["OPENAI_API_KEY"]
        ):
            st.session_state["OPENAI_API_KEY"] = st.session_state["input_OPENAI_API_KEY"]
            if st.session_state["pdfquery"] is not None:
                st.warning("Please, upload the files again.")
            st.session_state["messages"] = []
            st.session_state["user_input"] = ""
            st.session_state["pdfquery"] = PDFQuery(st.session_state["OPENAI_API_KEY"])
    
    st.subheader("上传一个文档")

    st.file_uploader(
        "上传一个文档",
        type=['pdf'],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
        disabled=not is_openai_api_key_set()
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", disabled=not is_openai_api_key_set(), on_change=process_input)

    st.divider()
    st.markdown("Source code: [Github](https://github.com/JiYueBo666/chatPDF)")
if __name__=="__main__":
    main()