from bs4 import BeautifulSoup
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

class DocReader():

    docs_retriever = None
    t_docs = None

    def __init__(self, main_model, st_model='mixedbread-ai/mxbai-embed-large-v1', url=''):
        self.embed_model = HuggingFaceEmbeddings(model_name=st_model)
        self.url = url
        self.loader = AsyncChromiumLoader([url])
        self.model = main_model

    def create_db(self, docs):
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embed_model,
            )

        self.docs_retriever = MultiQueryRetriever.from_llm(
            llm=self.model,
            retriever=vector_store.as_retriever(),
            )

    # Experimental
    def extract_links(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        links = [link.get('href') for link in soup.find_all('a')]
        return links

    async def crawl_and_load(self, url, visited=None):
        if visited is None:
            visited = set()

        visited.add(url)
        html_docs = await self.loader.aload()
        #content = html_docs[0].page_content
        #links = self.extract_links(html_docs)

        return html_docs
