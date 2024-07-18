# -*- coding: utf-8 -*-


# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install necessary packages
!pip install openai PyPDF2
!pip install langchain
!pip install faiss-cpu
!pip install langchain-community
!pip install unstructured
!pip install langchain-openai

# Set the OpenAI API key
import os
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
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vectorStore_openai = FAISS.from_documents(docs, embeddings)

# Save the FAISS vector store
with open("faiss_store_openai.pkl", 'wb') as f:
    pickle.dump(vectorStore_openai, f)

# Load the FAISS vector store
with open("faiss_store_openai.pkl", 'rb') as f:
    vectorStore = pickle.load(f)

# Set up the RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0, model_name="text-davinci-003")
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore.as_retriever())

# Test the chain
#result = chain({"question": "How do I know if I have depression?"}, return_only_outputs=True)
#print(result)