import os
import asyncio
import uuid
import logging
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema import Document
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO)

class DocReader:
    """ Document creator and reader class """
    def __init__(self, main_model, st_model='sentence-transformers/all-mpnet-base-v2', base_url=''):
        self.embed_model = HuggingFaceEmbeddings(model_name=st_model)
        self.base_url = base_url
        self.model = main_model
        self.visited = set()
        self.docs_retriever = None
        # Disable telemetry for ChromaDB
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

    def create_db(self, docs):
        """ Create vector database and retriever """
        if not docs:
            logging.warning("No documents to create database from.")
            return

        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        ids = [str(uuid.uuid4()) for _ in docs]

        logging.info("Creating database with %d documents.", len(docs))

        vector_store = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embedding=self.embed_model,
        )

        self.docs_retriever = vector_store.as_retriever()

    def extract_content(self, html):
        """ Extract content without webpage elements """
        soup = BeautifulSoup(html, 'html.parser')
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        # Extract text content
        return soup.get_text(separator=' ', strip=True)

    def extract_links(self, html, url):
        """ Extract links from the webpage """
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            if full_url.startswith(self.base_url) and full_url not in self.visited:
                links.append(full_url)
        return links

    async def crawl_and_load(self, url, page):
        """ Web Crawler """
        if url in self.visited:
            return []

        self.visited.add(url)

        try:
            await page.goto(url, wait_until="networkidle")
            content = await page.content()
        except Exception as e:
            logging.error("Error loading %s: %s", url, str(e))
            return []

        text_content = self.extract_content(content)
        links = self.extract_links(content, url)

        doc = Document(page_content=text_content, metadata={"source": url})

        logging.info("Crawled %s, found %d links", url, len(links))

        tasks = [self.crawl_and_load(link, page) for link in links]
        child_docs = await asyncio.gather(*tasks)

        return [doc] + [d for sublist in child_docs for d in sublist]

    async def process_documentation(self):
        """ Entrypoint for reading online documentation """
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            all_docs = await self.crawl_and_load(self.base_url, page)

            await browser.close()

        logging.info("Total documents crawled: %d", len(all_docs))
        if all_docs:
            self.create_db(all_docs)
        else:
            logging.warning("No documents were crawled.")
        return all_docs

    async def query(self, query: str, num_results: int = 3):
        """ Query vector database and retrieve results """
        if not self.docs_retriever:
            raise ValueError("Document retriever has not been initialized.\
                             Call process_documentation first.")

        relevant_docs = self.docs_retriever.invoke(input=query)
        return relevant_docs[:num_results]
