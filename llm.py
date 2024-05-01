#!/usr/bin/python3
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

## Download the GGUF model
model_name = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
model_file = "Meta-Llama-3-8B-Instruct.Q3_K_S.gguf"
model_path = hf_hub_download(model_name, filename=model_file)

## Instantiate model from downloaded file
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,        # Number of model layers to offload to GPU
)

## Generation kwargs
generation_kwargs = {
    "max_tokens":20000,
    "stop":["</s>"],
    "echo":True, # Echo the prompt in the output
    "temperature":0.3
}

## Chat session loop
while True:
    user_prompt = input("You: ")
    if user_prompt.lower() in ["exit", "quit", "bye"]:
        print("Chat session ended. Goodbye!")
        break
    res = llm(user_prompt, **generation_kwargs) # Res is a dictionary
    generated_text = res["choices"][0]["text"]
    print(f"Model: {generated_text}")
