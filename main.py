from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from diffusers import StableDiffusionPipeline
import torch
import uuid

app = FastAPI()

# Load the Stable Diffusion Model
model_id = "nitrosocke/Ghibli-Diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Move model to GPU

@app.get("/")
def read_root():
    return {"message": "Ghibli Style AI API is running!"}

@app.post("/generate/")
async def generate_image(
    prompt: str,
    num_inference_steps: int = Query(50, ge=10, le=100, description="Number of inference steps (10-100)")
):
    """
    Generate an image using the Ghibli AI model.
    Users can specify the number of inference steps (default: 50).
    """
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=9).images[0]
    
    # Save image with a unique name
    output_path = f"generated_{uuid.uuid4().hex}.png"
    image.save(output_path)

    return FileResponse(output_path, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
