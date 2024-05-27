""" Llama4U """
import sys
from os import devnull
from contextlib import contextmanager,redirect_stderr
from termcolor import colored
from huggingface_hub import hf_hub_download
import llama_cpp
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.chains.conversation.base import ConversationChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate
)
from input.input import parse_arguments

LLAMA4U_STR = 'Llama4U'

class Llama4U():
    """ Llama4U """

    # Model config parameters
    model_kwargs = {
        "n_gpu_layers": -1,
        "logits_all": True,
        'split_mode':llama_cpp.LLAMA_SPLIT_MODE_LAYER,
        'vocab_only': False,
        'use_mmap': True,
        'use_mlock': False,
        'kv_overrides': None,
        'seed': llama_cpp.LLAMA_DEFAULT_SEED,
        'n_ctx': 2048,
        'n_batch': 512,
        'rope_scaling_type': llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        'pooling_type': llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
        'rope_freq_base': 0.0,
        'rope_freq_scale': 0.0,
        'yarn_ext_factor':-1.0,
        'yarn_attn_factor': 1.0,
        'yarn_beta_fast': 32.0,
        'yarn_beta_slow': 1.0,
        'yarn_orig_ctx': 0,
        'embedding': False,
        'offload_kqv': True,
        'flash_attn': False,
        'last_n_tokens_size': 64,
        'lora_scale': 1.0,
        'numa': False,
        'chat_format': 'llama-2',
        'chat_handler': None,
        'verbose':True,
    }

    # Chat config parameters
    chat_kwargs = {
        'temperature': 0.2,
        'top_p': 0.95,
        'top_k': 40,
        'min_p': 0.05,
        'typical_p': 1.0,
        'max_tokens': None,
        'echo': False,
        'presence_penalty':0.0,
        'frequency_penalty':0.0,
        'repeat_penalty':1.1,
        'tfs_z':1.0,
        'mirostat_mode': 0,
        'mirostat_tau': 5.0,
        'mirostat_eta': 0.1,
        'logprobs': True,
        #'top_logprobs': 1,
    }

    # Define the human message template
    human_template = HumanMessagePromptTemplate.from_template(
        "{history}<|eot_id|>\n\n{input}<|eot_id|>"
        )

    # Combine the templates into a chat prompt template
    chat_template = ChatPromptTemplate.from_messages([human_template])

    def __init__(self,
                 hf_repo_id,
                 model_filename
                 ):
        if hf_repo_id is None:
            self.hf_repo_id='PawanKrd/Meta-Llama-3-8B-Instruct-GGUF'
        if model_filename is None:
            model_filename='llama-3-8b-instruct.Q3_K_M.gguf'
        self.model_path = hf_hub_download(repo_id=self.hf_repo_id, filename=model_filename)

        # Initialize LLM
        self.llm = LlamaCpp(
            model_path=self.model_path,
            **self.model_kwargs,
        )

        # Initialize Conversation "Chain"
        # using our LLM, chat template and config params
        self.conversation_chain = ConversationChain(
                llm=self.llm,
                prompt=self.chat_template,
                memory=ConversationBufferMemory(),
                llm_kwargs=self.chat_kwargs,
            )

    def process_user_input(self):
        """ Get input from stdout """
        print(colored('>>> ', 'yellow'), end="")
        user_prompt = input()
        if user_prompt.lower() in ["exit", "quit", "bye"]:
            print(colored(f'{LLAMA4U_STR}: =====', 'yellow'))
            print("Chat session ended. Goodbye!")
            sys.exit(0)
        return user_prompt

    def start_chat_session(self, query=""):
        """ Chat session loop """
        my_messages=""
        stop_next_iter = False
        for _ in range(50):
            if stop_next_iter:
                break

            # User's turn
            if not query:
                my_messages = self.process_user_input()
            else:
                my_messages = query
                stop_next_iter = True

            # AI's turn
            response = self.conversation_chain.predict(input=my_messages)
            print(response.strip())

@contextmanager
def suppress_stderr(verbose):
    """A context manager that redirects stderr to devnull based on verbose selection """
    if verbose <= 0:
        with open(devnull, 'w', encoding='utf-8') as fnull:
            with redirect_stderr(fnull) as err:
                yield err
    else:
        yield ()

def main():
    """ Pip Package entrypoint """
    args = parse_arguments()
    if args.verbose:
        verbose = args.verbose
    else:
        verbose = 0

    with suppress_stderr(verbose):
        llama4u = Llama4U(args.repo_id, args.filename)
        llama4u.start_chat_session(args.query)

if __name__ == '__main__':
    main()
