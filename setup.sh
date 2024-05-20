#!/bin/bash

if [ "$1" != "cpu" ]; then
    # nvidia GPU setup for Ubuntu 22.04
    curl -fSsL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor | sudo tee /usr/share/keyrings/nvidia-drivers.gpg > /dev/null 2>&1
    echo 'deb [signed-by=/usr/share/keyrings/nvidia-drivers.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' | sudo tee /etc/apt/sources.list.d/nvidia-drivers.list
    sudo apt update
    sudo apt install cuda-toolkit-12-4
    export PATH=/usr/local/cuda-12/bin:~/.local/bin:${PATH}
    export CUDACXX=$(which nvcc)
    if -z $CUDACXX; then
        echo "nvcc not found in PATH."
        exit /b 1
    fi
    echo $CUDACXX && $CUDACXX --version
fi

# project dependencies
sudo apt install python3-pip
pip install -r requirements.txt
