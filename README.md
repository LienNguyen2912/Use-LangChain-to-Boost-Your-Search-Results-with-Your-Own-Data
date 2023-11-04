# Use LangChain to Boost Search Results with Our Own Data
## The issues:
- ChatGPT cannot access private data
- Manually copying extensive data into chatGPT is tedious and limited by the prompt length
- Private data may consist of updated news or personal schedules, and it may change frequently.
## Langchain is the solution:
- Load files by `file_uploader`
  ```
  uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True)
  for uploaded_file in uploaded_files:
            # To read file as string:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            data += st.text_area(uploaded_file.name + " content", stringio.read())
  ```
- Vectorize data by `OpenAIEmbeddings`, store in `FAISS` database, and search by using the FAISS vector search API. Why?
    - Accuracy: Vectorized data can be searched more accurately than text data. This is because vectorized data represents the meaning of the text in a mathematical way, which makes it easier for LangChain to identify similar documents.
    - Speed: Vectorized data can be searched much faster than text data.
    - Reduce costs: use the OpenAI API to search for text, you are charged based on the number of tokens in the prompt. Therefore, searching from the most relevant vectors retrieved will result in significant cost savings.
     ```
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.',','], chunk_size= 1000)
        docs = text_splitter.split_text(data)
        ...
        #Vectorize and save into DB
        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_texts(docs, embeddings)
      
        #save the FAISS into a pickle file
        file_path = "faiss_store_openai.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(vectors, f)
        ...
        llm = OpenAI(temperature=0.9, max_tokens=500)
        file_path = "faiss_store_openai.pkl"
        if os.path.exists(file_path) and (os.path.getsize(file_path) > 0):
          with open(file_path, "rb") as f:
              vectorstore = pickle.load(f)
              chain = RetrievalQA.from_llm(llm= llm, retriever=vectorstore.as_retriever())
              result = chain(query, return_only_outputs=True)
     ```

When you provide data about LangChain in the `LangChainNews.txt` file, the OpenAI model can respond to relevant questions.</br>
<img width="608" alt="image" src="https://github.com/LienNguyen2912/Use-LangChain-to-Boost-Your-Search-Results-with-Your-Own-Data/assets/73010204/2ea0ea46-d406-4031-ba97-b0dc35f97d8e"></br>
Compare with no additional information, it should be </br>
<img width="410" alt="image" src="https://github.com/LienNguyen2912/Use-LangChain-to-Boost-Your-Search-Results-with-Your-Own-Data/assets/73010204/b526b8fb-62fb-47b8-88d9-951dfcfab3bc"></br>
## How to run:
- Execute `pip install -r requirement.txt` to install necessary libraries.
- Insert your OpenAI key in the `.evn` file
- Execute `streamlit run .\main.py` , `http://localhost:8501/` will be launched :laughing: :yum:
