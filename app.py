
import os
import tempfile
import streamlit as st
from chatbot import load_and_chunk, build_vector_store, build_chain, ask
 
st.set_page_config(page_title="PageSage", page_icon="📄", layout="centered")
 

st.markdown("""
<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] },
  svg: { fontCache: 'global' }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
""", unsafe_allow_html=True)
 
st.title(" PageSage")
st.caption("Upload a PDF and ask anything about it — including diagrams and numericals.")
 
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
 
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
 
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
 
                    chunks = load_and_chunk(tmp_path)
                    vector_store = build_vector_store(chunks)
                    st.session_state.retriever = build_chain(vector_store)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    os.unlink(tmp_path)
                    st.success(f"✅ Ready! {len(chunks)} chunks indexed.")
 
                except ValueError as e:
                    st.error(f"❌ {e}")
                    st.session_state.pdf_name = None
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                    st.session_state.pdf_name = None
 
    if st.session_state.pdf_name:
        st.info(f"📂 Active: **{st.session_state.pdf_name}**")
 
    if st.button(" Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.retriever = None
        st.session_state.pdf_name = None
        st.rerun()
 
if st.session_state.retriever is None:
    st.info(" Upload a PDF from the sidebar to get started.")
else:
    for q, a, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.markdown(a)
            if sources:
                st.caption(f"📖 Source pages: {sources}")
 
    user_question = st.chat_input("Ask a question about your PDF...")
 
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = ask(
                        st.session_state.retriever,
                        user_question,
                        st.session_state.chat_history
                    )
                    st.markdown(answer)
                    if sources:
                        st.caption(f"📖 Source pages: {sources}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    answer, sources = "Error occurred.", []
 
        st.session_state.chat_history.append((user_question, answer, sources))
 
