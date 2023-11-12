import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline


# Load the SDXL refiner model locally or online.
def load_sdxl_refiner_model(model_type, model_load_type):
    if model_type == "server":
        model_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
    elif model_type == "local":
        model_path = "./models/stable-diffusion-xl-refiner-1.0"

    if model_load_type == "pretrained":
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
    elif model_load_type == "single":
        pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
            "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")

    return pipeline
