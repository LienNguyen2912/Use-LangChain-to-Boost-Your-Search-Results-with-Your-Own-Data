import os
import pickle
import time
import streamlit as st
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Get openAI API key
load_dotenv()
DEBUG = False

st.cache_data.clear()
#main_placeholder = st.empty()
st.title ("Search from your specified sources")
st.sidebar.title(("Files to use for query"))
query = st.text_input("Question: ",value="")

uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True)
process_data_clicked = st.sidebar.button("Process data")

if process_data_clicked:
    query = ""
    main_placeholder = st.empty()
    time.sleep(1)
    if uploaded_files is not None and len(uploaded_files) > 0:
        # Load data from file
        data = ""
        main_placeholder.text("Data is reading ..")
        for uploaded_file in uploaded_files:
            # To read file as string:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            data += st.text_area(uploaded_file.name + " content", stringio.read())
        main_placeholder.subheader(f'Finish loading {len(data)} characters.')
        
        # split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.',','], chunk_size= 1000)
        main_placeholder.text("Data is splitting")
        # st.write(data)
        docs = text_splitter.split_text(data)
        main_placeholder.subheader(f'Finish splitting into {len(docs)} chunks')
        if DEBUG:
            for doc in docs:
                st.write(doc)
                st.divider()

        # vectorize and save into DB
        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_texts(docs, embeddings)
        main_placeholder.subheader("Building DB of vectors..")
        time.sleep(2)

        # save the FAISS into a pickle file
        file_path = "faiss_store_openai.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(vectors, f)
        main_placeholder.subheader("Finish building DB of vectors")
    else: 
        main_placeholder.subheader(f'No input files for querying')


if query:
    llm = OpenAI(temperature=0.9, max_tokens=500)
    file_path = "faiss_store_openai.pkl"
    if os.path.exists(file_path) and (os.path.getsize(file_path) > 0):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_llm(llm= llm, retriever=vectorstore.as_retriever())
            result = chain(query, return_only_outputs=True)
            st.write(result["result"])
    else:
        st.write("There is no additional source file using for query", llm(query))

    



