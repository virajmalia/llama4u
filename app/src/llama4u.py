""" Llama4U """
import asyncio
import argparse
import importlib.metadata
from termcolor import colored
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec

LLAMA4U_STR = 'Llama4U'

def parse_arguments():
    """ parse input arguments """
    version = importlib.metadata.version('Llama4U')
    parser = argparse.ArgumentParser(description=f'Llama4U v{version}')
    parser.add_argument('-q', '--query', type=str, required=False, help='Single Query')
    parser.add_argument('-v', '--verbose', type=int, required=False, help='Enable verbose output')
    return parser.parse_args()

class Llama4U():
    """ Llama4U """

    system_prompt = """You are a helpful AI assistant named Llama4U."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ('human', "{input}"),
        ]
    )

    llm = ChatOllama(model='llama3')
    runnable = prompt | llm

    doc_url = None

    store = {}

    def __init__(self):
        # Initialize LLM chat chain
        self.with_msg_history = RunnableWithMessageHistory(
            runnable=self.runnable, # type: ignore
            get_session_history=self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="doc_url",
                    annotation=str,
                    name="Web URL",
                    description="Base URL of the webpage to use as context.",
                    default="",
                    is_shared=True,
                ),
                ],
            )

    def get_session_history(self, doc_url):
        """ Get session history from session_id """
        self.doc_url = doc_url
        self.store[0] = ChatMessageHistory()
        return self.store[0]

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
