from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles  # Import StaticFiles
from app.middleware.processPrompt import promptResponse
import torch
print(torch.cuda.is_available())
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import os
import numpy as np

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="app/templates")

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def process_form(request: Request, prompt: str = Form(...)):
    # Process the prompt and get the response
    newResponse = promptResponse(prompt)
    
    # Generate video based on the prompt
    video_frames = pipe(prompt, num_inference_steps=25, num_frames=200).frames
    video_frames_np = [np.array(frame) for frame in video_frames]
    video_frames_np = np.concatenate(video_frames_np, axis=0)
    video_path = export_to_video(video_frames_np)

    # Ensure the static directory exists
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    # Save the video to the static directory
    static_video_path = os.path.join(static_dir, "generated_video.mp4")
    with open(video_path, "rb") as source_file:
        video_data = source_file.read()

    with open(static_video_path, "wb") as f:
        f.write(video_data)
    
    # Return the response with the video path adjusted to not duplicate the '/static' prefix
    return templates.TemplateResponse("form.html", {"request": request, "response": newResponse, "video_path": "generated_video.mp4"})
