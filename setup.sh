# nvidia GPU setup for Ubuntu 22.04
curl -fSsL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor | sudo tee /usr/share/keyrings/nvidia-drivers.gpg > /dev/null 2>&1
echo 'deb [signed-by=/usr/share/keyrings/nvidia-drivers.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' | sudo tee /etc/apt/sources.list.d/nvidia-drivers.list
sudo apt update
sudo apt install cuda-toolkit-12-4
export PATH=/usr/local/cuda-12/bin:~/.local/bin:${PATH}
which nvcc && nvcc --version

# project dependencies
pip install -r requirements.txt
export CUDACXX=/usr/local/cuda-12/bin/nvcc && CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

# runtime dependencies
export LLAMA_CPP_LIB=$(python3 -m site --user-site)/llama_cpp/libllama.so
export CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:${LD_LIBRARY_PATH}

# GPU device verification
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 2> /dev/null
