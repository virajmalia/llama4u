#!/bin/env python3
from duckduckgo_search import DDGS
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import contains
from re import search
import json
import sys

# Input query
input = sys.argv[1]

# Get DuckDuckGo search results
results = DDGS().text(input, max_results=3)
urls = []
for res in results:
    urls.append(res.get('href'))

# Create an instance of WebBaseLoader
loader = WebBaseLoader(urls)
data = loader.load()

# Convert the loaded data to a list of dictionaries
# output = [
#     {
#         "url": doc.metadata["source"],
#         "content": doc.page_content,
#     }
#     for doc in data
# ]

# Save the output to a JSON file
# with open("scraped_data.json", "w", encoding="utf-8") as file:
#     json.dump(output, file, ensure_ascii=False, indent=4)

# Split the output to keep small sized chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# Create a VectorDB from the DDG search results
embedding = HuggingFaceEmbeddings(model_name='multi-qa-MiniLM-L6-cos-v1')
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

# Use the vectorDB as a retriever
retriever = vectordb.as_retriever()

# Query the vectorDB
docs = retriever.invoke(input)
print(docs[0].page_content)

# Test the vectordb
# documents = vectordb.similarity_search(input, k=5)
# for doc in documents:
#     print(doc.page_content)
