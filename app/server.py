#!/usr/bin/env python
""" Llama4U Server """
from typing import List, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.pydantic_v1 import Field
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langserve import CustomUserType
from langserve.server import add_routes
from app.src.llama4u import Llama4U

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
    chat: List[Union[HumanMessage, AIMessage]] = \
        Field(..., extra={"widget": {"type": "chat", "input": "chat"}})

def format_input(input_data: ChatHistory):
    """ Format input from the langserve UI """
    chat_history = input_data.chat

    msg_history = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            msg_history.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            msg_history.append({"role": "assistant", "content": message.content})

    input_data = {"input": msg_history}

    return input_data

def format_output(response_data):
    """ Format output from the Chain for the langserve UI """
    return response_data.content

llama4u = Llama4U()
input_formatter = RunnableLambda(format_input)
chat_model = input_formatter | llama4u.with_msg_history
add_routes(
    app,
    chat_model.with_types(input_type=ChatHistory, output_type=ChatHistory),
    config_keys=["configurable"],
    path="/llama4u",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost")
