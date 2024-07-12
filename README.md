# FastAPI Application

This is a FastAPI application that demonstrates basic usage.

This was template was created by [Sola Akinbode](https://github.com/OAkinbode)

We have altered the template to create videos using this [Hugging Face Algorithm](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b)

## Prerequisites

Must have a CUDA compatible GPU (NVIDIA) and have CUDA installed on your system.

You can find the installation [here](https://developer.nvidia.com/cuda-downloads)

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/OAkinbode/ET-AIModel.git
cd ET-AIModel

```

### 2. Install all dependencies using either of the following:

pip install -r requirements.txt

or

pip3 install -r requirements.txt

pip install diffusers transformers accelerate torch

pip install git+https://github.com/huggingface/diffusers transformers accelerate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install opencv-python

### 3. Run the application using for mac

uvicorn app.main:app --reload

### 3. Run the application using for windows

python.exe -m uvicorn.main app.main:app --reload

### 4. Use application in browser

visit http://localhost:8000
