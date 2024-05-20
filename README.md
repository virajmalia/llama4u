# The Llama4U Project
[![Python application](https://github.com/virajmalia/llama4u/actions/workflows/CI.yml/badge.svg)](https://github.com/virajmalia/llama4u/actions/workflows/CI.yml)

## Vision
Develop a free and open source, fully-featured AI solution with agents.

## Current motivations for feature-set
- Perplexity AI
- ChatGPT/GPT4o

## Rule
- APIs that have usage limitations or require keys to be registered with an online account are not permitted to be added to this project.

## System requirements
- Nvidia GPU (>=8G VRAM)
- Ubuntu 22.04
- Works on WSL2 with nvidia CUDA

## Steps to run
1. `./setup.sh`
2. `./src/llama4u.py`

    Default model: https://huggingface.co/PawanKrd/Meta-Llama-3-8B-Instruct-GGUF/blob/main/llama-3-8b-instruct.Q3_K_M.gguf

Full CLI: `./src/llama4u.py -r <repo_id> -f <filename> -q <query>`

## Description
Llama4U is an AI assistant developed using [LlamaCPP][1], [LangChain][2] and [Llama3][3]. A completely free AI solution that can be hosted locally, while providing online capabilities in a responsible and user-controllable way.

## Credits
- Meta, for the open source Llama models
- HuggingFace community
- LlamaCPP and llama-cpp-python communities
- LangChain community


[1]: https://github.com/abetlen/llama-cpp-python
[2]: https://python.langchain.com/v0.1/docs/get_started/introduction/
[3]: https://huggingface.co/blog/llama3
