from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.middleware.processPrompt import promptResponse
import torch
print(torch.cuda.is_available())
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import os

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="app/templates")

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
    video_path = export_to_video(video_frames)
    
    # Save the video to the static directory
    static_video_path = os.path.join("static", "generated_video.mp4")
    with open(static_video_path, "wb") as f:
        f.write(video_path)
    
    # Return the response with the video path
    return templates.TemplateResponse("form.html", {"request": request, "response": newResponse, "video_path": "generated_video.mp4"})
