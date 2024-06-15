#!/usr/bin/env python
""" Llama4U Server """
from typing import List, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.pydantic_v1 import Field
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_transformers.html2text import Html2TextTransformer
from langserve import CustomUserType
from langserve.server import add_routes
from app.src.llama4u import Llama4U
from app.src.parsers.documents import DocReader

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class ChatHistory(CustomUserType):
    """ Playground Chat Widget """
    chat_history: List[Union[HumanMessage, AIMessage]] = \
        Field(..., extra={"widget": {"type": "chat", "input": "chat_history"}})

async def get_response_from_docs(url):
    """ Get a response from a URL page content """
    doc_reader = DocReader(llama4u.llm, url=url)
    crawled_data = await doc_reader.crawl_and_load(url)

    h2t = Html2TextTransformer()
    t_docs = h2t.transform_documents(crawled_data)
    doc_reader.create_db(t_docs)
    return doc_reader.docs_retriever.invoke( # type: ignore
        input='Read each word carefully and \
            find the relevance with programming and APIs. \
            Summarize the document such that it can be used \
            as a context for future conversations.')

async def format_input(input_data: ChatHistory, config):
    """ Format input from the langserve UI """
    chat_history = input_data.chat_history
    response = None
    md = config.get('metadata')
    if md:
        url = md.get('doc_url')
        if url:
            response = await get_response_from_docs(url)

    msg_history = []
    if response and len(msg_history) == 0:
        # Append the response from docs query to message history
        # Only do it the first time
        msg_history.append(
            HumanMessage(
                content=f"Use this document \
                    as context for the rest of our conversation: \
                    {response[0].page_content}"
            )
        )

    for message in chat_history:
        if isinstance(message, HumanMessage):
            msg_history.append(HumanMessage(content=message.content))
        elif isinstance(message, AIMessage):
            msg_history.append(AIMessage(content=message.content))

    input_data = {"input": msg_history, "chat_history": msg_history} # type: ignore

    return input_data

llama4u = Llama4U()

input_formatter = RunnableLambda(format_input)
chat_model = input_formatter | llama4u.with_msg_history

add_routes(
    app,
    chat_model.with_types(
        input_type=ChatHistory,
        output_type=ChatHistory,
        ).with_config(
            configurable={"doc_url": "doc_url"}),
    config_keys=["configurable", "doc_url"],
    path="/llama4u",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost")
