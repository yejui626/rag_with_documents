import streamlit as st
import time
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import format_document

def init_session(_session_key, _chat_name, clear: bool =False):
    if not chat_inited or clear:
        st.session_state[_session_key] = {}
        time.sleep(0.1)
        reset_history(_chat_name)

def chat_inited(_session_key):
    return _session_key in st.session_state.keys()

def reset_history(_session_key,_chat_name, name=None):
    if not chat_inited:
        st.session_state[_session_key] = {}

    name = name or _chat_name
    st.session_state[_session_key][name] = []

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


