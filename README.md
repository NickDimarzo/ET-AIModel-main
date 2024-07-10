# FastAPI Application

This is a FastAPI application that demonstrates basic usage.

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
