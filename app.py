import streamlit as st
from health_ir import DOCS, SearchEngine

st.set_page_config(page_title="Healthcare Information Search")
st.title("Healthcare Information Search System")
st.caption("For information retrieval demonstration only. Not medical advice.")

@st.cache_resource
def load():
    return SearchEngine(DOCS)
eng = load()

q = st.text_input("Enter your health query")
c1, c2, c3 = st.columns(3)
model = c1.selectbox("Model", ["tfidf", "bm25"])
expand = c2.checkbox("Synonym expansion")
k = c3.slider("Results", 1, 10, 5)

if q:
    st.write("Processed query terms:", eng.query_terms(q, expand))
    results = eng.search(q, model, expand, k)
    if not results:
        st.warning("No matching documents.")
    for d, s in results:
        st.subheader(f"{d['title']}  (score {s:.3f})")
        st.caption(d["category"])
        st.write(d["text"])