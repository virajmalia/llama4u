""" Llama4U """
import asyncio
from termcolor import colored
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.src.input.input import parse_arguments

LLAMA4U_STR = 'Llama4U'

class Llama4U():
    """ Llama4U """

    system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, \
            formulate a response that is clear and understandable by an 18yo human."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOllama(model='llama3')
    runnable = prompt | llm

    store = {}

    def __init__(self):
        # Initialize LLM chat chain
        self.with_msg_history = RunnableWithMessageHistory(
            runnable=self.runnable,
            get_session_history=self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            )

    def get_session_history(self, session_id):
        """ Get session history from session_id """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    async def chat_session(self):
        """ Chat session with history """
        while True:
            # Get input
            print(colored('>>> ', 'yellow'), end="")
            user_prompt = input()

            # Redirect search queries
            if user_prompt.startswith("/search"):
                search_results = DuckDuckGoSearchRun().run(user_prompt.replace("/search", ""))
                user_prompt = \
                f"Summarize the following search results as if you are answering:{search_results}"

            # Invoke chain
            response = self.with_msg_history.invoke(
                {"input": user_prompt},
                config={"configurable": {"session_id": "abc123"}},
                )
            print(response.content)

    async def dispatch(self, query=""):
        """ Dispatch query """
        if query:
            response = self.llm.invoke(input=query)
            query=""
            print(response.content)
        else:
            await self.chat_session()

def main():
    """ Pip Package entrypoint """
    args = parse_arguments()

    llama4u = Llama4U()
    asyncio.run(llama4u.dispatch(args.query))

if __name__ == '__main__':
    main()
