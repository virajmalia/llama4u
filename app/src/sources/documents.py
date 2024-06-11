from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import Settings

class DocReader():
    def __init__(self, main_model, st_model='mixedbread-ai/mxbai-embed-large-v1', directory_path='/mnt/c/Users/viraj/Documents/ai_db/'):
        self.embed_model = HuggingFaceEmbeddings(model_name=st_model)

        self.directory_path = directory_path
        reader = SimpleDirectoryReader(directory_path)
        docs = reader.load_data()

        Settings.llm = main_model
        Settings.context_window = 8000
        service_context = ServiceContext.from_defaults(embed_model=self.embed_model)
        self.index = VectorStoreIndex.from_documents(docs, service_context=service_context)

    def get_query_engine(self, model):
        query_engine = self.index.as_query_engine(model)
        return query_engine
