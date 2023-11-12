import sys
import argparse, os
import cv2
import torch
from PIL import Image
from load_sdxl_base_model import load_sdxl_base_model
from load_sdxl_refiner_model import load_sdxl_refiner_model


MODEL_TYPE="server"
MODEL_LOAD_TYPE="pretrained"
MODEL="base"
OUTPUT_PATH="output"


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate an image based on a prompt")

    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True, 
        help="The prompt or prompts to guide the image generation",
        default="a professional photograph of an astronaut riding a horse"
    )
    parser.add_argument(
        "--prompt_2", 
        type=str, 
        required=False, 
        help="The prompt or prompts to be sent to the tokenizer_2 and text_encoder_2",
        default="a professional photograph of an astronaut riding a horse"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        required=False, 
        help="The height in pixels of the generated image. This is set to 1024 by default for the best results.",
        default=1024
    )
    parser.add_argument(
        "--width", 
        type=int, 
        required=False, 
        help="The width in pixels of the generated image. This is set to 1024 by default for the best results.",
        default=1024
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        required=False, 
        help="The int of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
        default=50
    )
    parser.add_argument(
        "--denoising_end", 
        type=float, 
        required=False, 
        help="When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be completed before it is intentionally prematurely terminated.",
        default=0.0
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        required=False, 
        help="Guidance scale as defined in Classifier-Free Diffusion Guidance.",
        default=5.0
    )
    parser.add_argument(
        "--negative_prompt", 
        type=str, 
        required=False, 
        help="The prompt or prompts not to guide the image generation.",
        default="nude adult"
    )
    parser.add_argument(
        "--negative_prompt_2", 
        type=str, 
        required=False, 
        help="The prompt or prompts not to guide the image generation to be sent to tokenizer_2 and text_encoder_2.",
        default="nude adult"
    )
    parser.add_argument(
        "--num_images_per_prompt", 
        type=int, 
        required=False, 
        help="The int of images to generate per prompt.",
        default=1
    )
    parser.add_argument(
        "--eta", 
        type=float, 
        required=False, 
        help="Corresponds to parameter eta (η) in the DDIM paper",
        default=0.0
    )
    parser.add_argument(
        "--generator", 
        type=torch.Generator,
         required=False, 
         help="One or a list of torch generator(s) to make generation deterministic.",
    )
    parser.add_argument(
        "--latents", 
        type=torch.FloatTensor,
         required=False, 
         help="Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation.",
    )
    parser.add_argument(
        "--prompt_embeds", 
        type=torch.FloatTensor,
         required=False, 
         help="Pre-generated text embeddings. Can be used to easily tweak text inputs, e.g. prompt weighting.",
    )
    parser.add_argument(
        "--negative_prompt_embeds", 
        type=torch.FloatTensor,
         required=False, 
         help="Pre-generated negative text embeddings. Can be used to easily tweak text inputs, e.g. prompt weighting.",
    )
    parser.add_argument(
        "--pooled_prompt_embeds", 
        type=torch.FloatTensor,
         required=False, 
         help="Pre-generated pooled text embeddings.",
    )
    parser.add_argument(
        "--negative_pooled_prompt_embeds", 
        type=torch.FloatTensor,
         required=False, 
         help="Pre-generated negative pooled text embeddings.",
    )
    parser.add_argument(
        "--output_type", 
        type=str, 
        required=False, 
        help="The output format of the generate image.",
        default="pil"
    )
    parser.add_argument(
        "--return_dict", 
        type=bool, 
        required=False, 
        help="Whether or not to return a ~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput instead of a plain tuple.",
        default=True
    )
    parser.add_argument(
        "--cross_attention_kwargs", 
        type=dict, 
        required=False, 
        help=" A kwargs dictionary that if specified is passed along to the AttentionProcessor as defined under self.processor in diffusers.models.attention_processor.",
    )
    parser.add_argument(
        "--guidance_rescale", 
        type=float, 
        required=False, 
        help="Guidance rescale factor proposed by Common Diffusion Noise Schedules and Sample Steps are Flawed guidance_scale is defined as φ in equation 16. of Common Diffusion Noise Schedules and Sample Steps are Flawed.",
        default=0.0
    )
    parser.add_argument(
        "--original_size", 
        type=tuple[int]
        , required=False,
         help="If original_size is not the same as target_size the image will appear to be down- or upsampled. original_size defaults to (height, width) if not specified.",
         default=(1024, 1024)
    )
    parser.add_argument(
        "--crops_coords_top_left", 
        type=tuple[int]
        , required=False,
         help="crops_coords_top_left can be used to generate an image that appears to be “cropped” from the position crops_coords_top_left downwards.",
         default=(0, 0)
    )
    parser.add_argument(
        "--target_size", 
        type=tuple[int]
        , required=False,
         help="For most cases, target_size should be set to the desired height and width of the generated image.",
         default=(1024, 1024)
    )
    parser.add_argument(
        "--negative_original_size", 
        type=tuple[int]
        , required=False,
         help="To negatively condition the generation process based on a specific image resolution.",
         default=(1024, 1024)
    )
    parser.add_argument(
        "--negative_crops_coords_top_left", 
        type=tuple[int]
        , required=False,
         help="To negatively condition the generation process based on a specific crop coordinates.",
         default=(0, 0)
    )
    parser.add_argument(
        "--negative_target_size", 
        type=tuple[int]
        , required=False,
         help="To negatively condition the generation process based on a target image resolution.",
         default=(1024, 1024)
    )
    parser.add_argument(
        "--callback_on_step_end", 
        type=callable, 
        required=False, 
        help="A function that calls at the end of each denoising steps during the inference.",
    )
    parser.add_argument(
        "--callback_on_step_end_tensor_inputs", 
        type=list, 
        required=False, 
        help="The list of tensor inputs for the callback_on_step_end function.",
    )

    args = parser.parse_args()
    return args


# Add a watermark to the image
def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def main(args):
    # Load stable diffusion xl model
    if MODEL == "base":
        model = load_sdxl_base_model(MODEL_TYPE, MODEL_LOAD_TYPE)
    elif MODEL == "refiner":
        model = load_sdxl_refiner_model(MODEL_TYPE, MODEL_LOAD_TYPE)

    # txt2img 
    image = model(args.prompt).images[0]

    return image
    # # Save the result
    # image.save(f"{OUTPUT_PATH}/{args.prompt}.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
