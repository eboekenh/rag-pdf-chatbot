import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

FAISS_INDEX_DIR = "faiss"


def build_vector_store(pdf_path: str) -> FAISS:
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
    os.makedirs(FAISS_INDEX_DIR)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)
    return vector_store


def build_chain(vector_store: FAISS):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks. "
         "Use the retrieved context below to answer the question. "
         "If the answer is not in the context, say you don't know.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# -- Streamlit UI --------------------------------------------------------------

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="centered")
st.title("📄 PDF RAG Chatbot")
st.caption("Upload a PDF, then ask anything about its contents.")

api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    api_key = st.text_input("Enter your Google Gemini API key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file and api_key:
    file_id = uploaded_file.name + str(uploaded_file.size)

    if st.session_state.get("file_id") != file_id:
        with st.spinner("Processing PDF and building vector index..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                vector_store = build_vector_store(tmp_path)
            finally:
                os.unlink(tmp_path)

        chain, retriever = build_chain(vector_store)
        st.session_state.file_id = file_id
        st.session_state.chain = chain
        st.session_state.retriever = retriever
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success(f"Ready! Indexed **{uploaded_file.name}**.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke({
                    "question": prompt,
                    "chat_history": st.session_state.chat_history,
                })
                sources = st.session_state.retriever.invoke(prompt)
            st.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    for doc in sources:
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Page {page + 1}:** {doc.page_content[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=answer),
        ])

elif uploaded_file and not api_key:
    st.warning("Please enter your Google Gemini API key above to continue.")
else:
    st.info("Upload a PDF file to get started.")
