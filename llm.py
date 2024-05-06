#!/bin/env python3
import sys
from llama_cpp import Llama

if len(sys.argv) < 2:
    print('Usage: ./llm.py <model_dir>')
    exit(1)
model_dir = sys.argv[1]
model_name='llama-3-8b-instruct.Q3_K_M.gguf'

local_model = Llama.from_pretrained(repo_id='PawanKrd/Meta-Llama-3-8B-Instruct-GGUF', filename=model_name, local_dir=model_dir)

## Instantiate model from downloaded file
llm = Llama(
    n_gpu_layers=-1,
    max_new_tokens=2048,
    model_path=model_dir+'/'+model_name
)

## Chat session loop
my_messages = [
    {"role": "system", "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."},
]

msg_count=0         # 1 msg_count includes 1 msg from the user and 1 msg from the aassistant
max_msg_count=50    # This limits the chat to max_msg_count*2 messages total.
while msg_count < max_msg_count:
    msg_count+=1
    print('You: =====')
    user_prompt = input()
    if user_prompt.lower() in ["exit", "quit", "bye"]:
        print("Chat session ended. Goodbye!")
        break
    my_messages.append({"role": "user", "content": user_prompt})

    response = llm.create_chat_completion(messages=my_messages)
    assistant_output = response["choices"][0]["message"]["content"]
    print('Assistant: =====')
    print(assistant_output)
