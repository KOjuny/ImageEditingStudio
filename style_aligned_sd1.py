import os
import argparse
import json
from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch
import numpy as np
import random
from models.stylealigned import sa_handler
import os

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion")
    parser.add_argument('--prompts_file', type=str, required=True, 
                        help="Path to the JSON file containing a list of prompts")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Directory to save the generated images")
    return parser.parse_args()

# Parse the arguments
args = parse_args()

# Load prompts from the JSON file
with open(args.prompts_file, 'r') as f:
    sets_of_prompts = json.load(f)

# Ensure the prompts are in a list format
if not isinstance(sets_of_prompts, list):
    raise ValueError("The JSON file must contain a list of prompts.")

# Define output directory
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False)
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    scheduler=scheduler
)
pipeline = pipeline.to("cuda")

setup_seed()
for prompt_set in sets_of_prompts:
    prompts = [f"{obj} {prompt_set['prompt']}" for obj in prompt_set['objects']]
    for i in range(1, len(prompts)):
        handler = sa_handler.Handler(pipeline)
        sa_args = sa_handler.StyleAlignedArgs(share_group_norm=True,
                                            share_layer_norm=True,
                                            share_attention=True,
                                            adain_queries=True,
                                            adain_keys=True,
                                            adain_values=False,)

        handler.register(sa_args)

        # Run StyleAligned
        images = pipeline([prompts[0],prompts[i]], generator=None).images
        
        # Save images
        output_folder = os.path.join(output_dir, prompt_set['prompt'])
        os.makedirs(output_folder, exist_ok=True)
        if not os.path.exists(os.path.join(output_folder, "generated_image_0.png")):
            images[0].save(os.path.join(output_folder, "generated_image_0.png"))
        img_path = os.path.join(output_folder, f"generated_image_{i}.png")
        images[1].save(img_path)
