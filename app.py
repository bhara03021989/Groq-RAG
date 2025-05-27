import streamlit as st 
import os 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time 

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
FAISS_INDEX_PATH = "faiss_index"
pdf_path = "Attention.pdf"

def vector_embeddings():

    if  os.path.exists(FAISS_INDEX_PATH):
        # load the existing FIASS index 
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.vectors = FAISS.load_local(FAISS_INDEX_PATH, st.session_state.embeddings , allow_dangerous_deserialization=True)
        st.write("FAISS index loaded from disk.")
    else:

        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/") # loader
        st.session_state.docs = st.session_state.loader.load() # now we will load 

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:100])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.session_state.vectors.save_local(FAISS_INDEX_PATH)
        st.write("FAISS index created and saved to disk.")


st.title("LangChain Groq")
llm = ChatGroq(groq_api_key = groq_api_key , model="gemma2-9b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context below. If the question cannot be answered based on the context,
    say 'I don't know'.
    {context}
    Question: {input}
    """
)

user_input  = st.text_input("Enter your question:")

if "embeddings_done" not in st.session_state :
    st.session_state.embeddings_done = False


if not st.session_state.embeddings_done:
    if st.button("Document Embeddings"):
        vector_embeddings()
        st.session_state.embeddings_done = True
        # Now i will disable the button so that it cannot be clicked again
        st.write("Document embeddings created and stored in session state.")
        #st.button("Document Embeddings", disabled=True , key="embed done button")
        st.write("You can now ask questions.")
else:
    st.button("Document Embeddings", disabled=True , key="embed done button")
    st.write("Document embeddings already created. You can ask questions now.")




if user_input :
    document_chain = create_stuff_documents_chain(llm , prompt)
    retriever = st.session_state.vectors.as_retriever() 
    retriever_chain = create_retrieval_chain(retriever , document_chain)
    start = time.process_time()
    response = retriever_chain.invoke({"input": user_input })
    end = time.process_time()

    time_taken = end - start
    st.write("Time taken to get the response: ", time_taken)
    st.write(response["answer"])
