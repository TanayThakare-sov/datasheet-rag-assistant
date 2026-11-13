"""Simple web UI.  Run with:  streamlit run app.py"""
import streamlit as st

from src.answer import answer_question
from src.ingest import ingest_folder
from src.retriever import Retriever

st.set_page_config(page_title="Datasheet Assistant", page_icon="💽")
st.title("Hard Drive Datasheet Assistant")
st.caption("Ask about specs in the datasheets, or compare drives. Answers cite the document and page.")

pdf_dir = st.sidebar.text_input("Folder with datasheet PDFs", "data/sample")
mode = st.sidebar.selectbox("Retrieval", ["tfidf", "hybrid"], help="hybrid needs sentence-transformers")
k = st.sidebar.slider("Passages used", 2, 10, 5)


@st.cache_resource(show_spinner="Reading datasheets...")
def load(folder, retrieval_mode):
    return Retriever(ingest_folder(folder, "table_aware"), mode=retrieval_mode)


try:
    retriever = load(pdf_dir, mode)
except Exception as e:
    st.error(f"Could not load datasheets: {e}")
    st.stop()

st.sidebar.write("Drives loaded:", ", ".join(retriever.docs))
question = st.text_input("Your question", placeholder="Compare the idle power of the Nimbus N8 and the Orion O12")

if question:
    out = answer_question(question, retriever, k=k)
    st.subheader("Answer")
    st.write(out["answer"])
    if out["sources"]:
        with st.expander("Sources"):
            for s in out["sources"]:
                st.markdown(f"**{s['doc']}, page {s['page']}**")
                st.text(s["text"])
