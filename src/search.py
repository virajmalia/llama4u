#!/bin/env python3
from duckduckgo_search import DDGS
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Search():
    """ Search class to perform online search using DuckDuckGo """
    def __init__(self, query_str):
        self.query = query_str
        self.embedding = HuggingFaceEmbeddings(model_name='multi-qa-MiniLM-L6-cos-v1')
        # Split the output to keep small sized chunks
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

        # Get DuckDuckGo search results
        results = DDGS().text(self.query, max_results=3)
        urls = []
        for res in results:
            urls.append(res.get('href'))

        # Create an instance of WebBaseLoader
        loader = WebBaseLoader(urls)
        self.data = loader.load()
        self.data_split = self.splitter.split_documents(self.data)

        # Create a VectorDB from the DDG search results
        self.vectordb = Chroma.from_documents(documents=self.data_split, embedding=self.embedding)

    def retrieve(self, db_query):
        """ Retrieve results of the search operation from the vectordb """
        # Use the vectorDB as a retriever
        retriever = self.vectordb.as_retriever()

        # Query the vectorDB
        docs = retriever.invoke(db_query)
        return docs[0].page_content
