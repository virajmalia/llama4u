#!/bin/env python3
from llama_cpp import Llama
from math import exp
from statistics import median
import sys

if len(sys.argv) < 2:
    print('Usage: ./llm.py <model_dir>')
    exit(1)
model_dir = sys.argv[1]
if len(sys.argv) == 3:
    single_query_mode = True
    query_str = sys.argv[2]
model_name='llama-3-8b-instruct.Q3_K_M.gguf'

local_model = Llama.from_pretrained(repo_id='PawanKrd/Meta-Llama-3-8B-Instruct-GGUF', filename=model_name, local_dir=model_dir)

# Instantiate model from downloaded file
llm = Llama(
    n_gpu_layers=-1,
    max_new_tokens=2048,
    model_path=model_dir + '/' + model_name,
    logits_all=True,
)

if single_query_mode:
    response = llm.create_chat_completion([{"role": "user", "content": query_str}],
                                          logprobs=True,
                                          top_logprobs=1,
                                          )
    if response:
        print(response)
        exit(0)
    else:
        print("Query failed")
        exit(1)

# Chat session loop
my_messages = [
    {"role": "system", "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."},
]

msg_count = 0          # 1 msg_count includes 1 msg from the user and 1 msg from the assistant
max_msg_count = 50     # This limits the chat to max_msg_count*2 messages total.
while msg_count < max_msg_count:
    msg_count+=1

    # User's turn
    print('You: =====')
    user_prompt = input()
    if user_prompt.lower() in ["exit", "quit", "bye"]:
        print("Chat session ended. Goodbye!")
        break
    my_messages.append({"role": "user", "content": user_prompt})

    # AI's turn
    response = llm.create_chat_completion(messages=my_messages,
                                          logprobs=True,
                                          top_logprobs=1
                                          )
    logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
    # Convert logprobs to probabilities
    probabilities = [exp(logprob) for logprob in logprobs]
    print(f'Assistant(Median Prob:{median(probabilities)}): =====')
    print(response["choices"][0]["message"]["content"])

#### Use cases ####
# 1. User input is vague or small
# 2. User input is clear, but the model cannot find an answer, and it is sure about it.
# 3. User input is clear, but the model cannot find an answer, and cannot determine so. (hallucinate?)
# 4. User input is very verbose.
