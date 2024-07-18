

import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install necessary packages
!pip install openai PyPDF2 langchain faiss-cpu langchain-community unstructured langchain-openai

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "key"

# Read the URL list from a text file
with open('/content/drive/MyDrive/urls.txt', 'r') as f:
    urls = [line.strip() for line in f]

# Verify the URLs were read correctly
print(urls)

# Load documents from URLs
from langchain.document_loaders import UnstructuredURLLoader
loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

# Split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

# Check the number of documents
print(f"Number of documents: {len(docs)}")

# Create FAISS vector store
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vectorStore_openai = FAISS.from_documents(docs, embeddings)

# Save the FAISS vector store
with open("/content/drive/MyDrive/faiss_store_openai.pkl", 'wb') as f:
    pickle.dump(vectorStore_openai, f)

# Load the FAISS vector store
def load_vector_store():
    with open("/content/drive/MyDrive/faiss_store_openai.pkl", 'rb') as f:
        vector_store = pickle.load(f)
    return vector_store

def main():
    st.title("ThinkHealth")

    user_question = st.text_area("Ask a question:")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if user_question:
        vector_store = load_vector_store()
        llm = OpenAI(temperature=0, model_name="text-davinci-003")
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())

        model_response = chain({"question": user_question})['answer']

        message = {'human': user_question, 'AI': model_response}
        st.session_state.chat_history.append(message)
        for chat in st.session_state.chat_history:
            st.write("You:", chat['human'])
            st.write("Chatbot:", chat['AI'])

if __name__ == "__main__":
    main()