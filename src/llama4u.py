""" Llama4U """
import importlib.metadata
import sys
import argparse
from math import exp
from statistics import median
from os import devnull
from contextlib import contextmanager,redirect_stderr
from termcolor import colored
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class Llama4U():
    """ Llama4U """
    def __init__(self,
                 hf_repo_id,
                 model_filename
                 ):
        if hf_repo_id is None:
            hf_repo_id="PawanKrd/Meta-Llama-3-8B-Instruct-GGUF"
        if model_filename is None:
            model_filename="llama-3-8b-instruct.Q3_K_M.gguf"
        model_path = hf_hub_download(repo_id=hf_repo_id, filename=model_filename)

        # Instantiate model from downloaded file
        self.llm = Llama(
            n_gpu_layers=-1,
            max_new_tokens=2048,
            model_path=model_path,
            logits_all=True,
        )

    def start_chat_session(self):
        """ Chat session loop """
        my_messages = [
            {"role": "system",
             "content": "A chat between a curious user and an artificial intelligence assistant. \
                The assistant gives helpful, and polite answers to the user's questions."},
        ]

        for _ in range(50):

            # User's turn
            print(colored('You: =====', 'yellow'))
            user_prompt = input()
            if user_prompt.lower() in ["exit", "quit", "bye"]:
                print(colored('Assistant(Median Prob:1.0): =====', 'yellow'))
                print("Chat session ended. Goodbye!")
                break
            my_messages.append({"role": "user", "content": user_prompt})

            # AI's turn
            response = self.llm.create_chat_completion(messages=my_messages,
                                                  logprobs=True,
                                                  top_logprobs=1,
                                                  )
            logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
            # Convert logprobs to probabilities
            probabilities = [exp(logprob) for logprob in logprobs]
            print(colored(f'Assistant(Median Prob:{median(probabilities)}): =====', 'yellow'))
            print(response["choices"][0]["message"]["content"])

    def single_query(self, query):
        """ Single Query Mode """
        response = self.llm.create_chat_completion([{"role": "user", "content": query}],
                                              logprobs=True,
                                              top_logprobs=1,
                                              )
        if response:
            logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
            # Convert logprobs to probabilities
            probabilities = [exp(logprob) for logprob in logprobs]
            print(f'Assistant(Median Prob:{median(probabilities)}): =====')
            print(response["choices"][0]["message"]["content"])
            sys.exit(0)
        else:
            print("Query failed")
            sys.exit(1)

@contextmanager
def suppress_stderr():
    """A context manager that redirects stderr to devnull"""
    with open(devnull, 'w', encoding='utf-8') as fnull:
        with redirect_stderr(fnull) as err:
            yield err

def parse_arguments():
    """ parse input arguments """
    version = importlib.metadata.version('Llama4U')
    parser = argparse.ArgumentParser(description=f'Llama4U v{version}')
    parser.add_argument('-r', '--repo_id', type=str, required=False, help='Repository ID')
    parser.add_argument('-f', '--filename', type=str, required=False, help='Filename')
    parser.add_argument('-q', '--query', type=str, required=False, help='Single Query')
    return parser.parse_args()

def main():
    """ Pip Package entrypoint """
    args = parse_arguments()
    repo_id = args.repo_id
    filename = args.filename

    with suppress_stderr():
        llama4u = Llama4U(repo_id, filename)

        if args.query:
            llama4u.single_query(args.query)
        else:
            llama4u.start_chat_session()

if __name__ == '__main__':
    main()
