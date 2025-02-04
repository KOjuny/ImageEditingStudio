from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm
import numpy as np
from collections.abc import Callable

import pdb

T = torch.Tensor
TN = T | None
InversionCallback = Callable[[StableDiffusionPipeline, int, T, dict[str, T]], dict[str, T]]


def _get_text_embeddings(prompt: str, tokenizer, text_encoder, device):
    # Tokenize text and get embeddings
    text_inputs = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_input_ids = text_inputs.input_ids

    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            output_hidden_states=True,
        )

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    return prompt_embeds, pooled_prompt_embeds


def _encode_text(model: StableDiffusionPipeline, prompt: str) -> T:
    tokenizer = model.tokenizer
    text_encoder = model.text_encoder
    device = model._execution_device
    prompt_embeds, _ = _get_text_embeddings(prompt, tokenizer, text_encoder, device)
    return prompt_embeds


def _encode_image(model: StableDiffusionPipeline, image: np.ndarray) -> T:
    model.vae.to(dtype=torch.float32)
    image = torch.from_numpy(image).float() / 255.
    image = (image * 2 - 1).permute(2, 0, 1).unsqueeze(0)
    latent = model.vae.encode(image.to(model.vae.device))['latent_dist'].mean * model.vae.config.scaling_factor
    model.vae.to(dtype=torch.float32)
    return latent


def _next_step(model: StableDiffusionPipeline, model_output: T, timestep: int, sample: T) -> T:
    timestep, next_timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = model.scheduler.alphas_cumprod[int(timestep)] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[int(next_timestep)]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def _get_noise_pred(model: StableDiffusionPipeline, latent: T, t: T, context: T, guidance_scale: float):
    latents_input = torch.cat([latent] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    return noise_pred


def _ddim_loop(model: StableDiffusionPipeline, z0, prompt, guidance_scale) -> T:
    all_latent = [z0]
    text_embedding = _encode_text(model, prompt)
    latent = z0.clone().detach()
    for i in tqdm(range(model.scheduler.num_inference_steps)):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
        noise_pred = _get_noise_pred(model, latent, t, text_embedding, guidance_scale)
        latent = _next_step(model, noise_pred, t, latent)
        all_latent.append(latent)
    return torch.cat(all_latent).flip(0)


def make_inversion_callback(zts, offset: int = 0) -> [T, InversionCallback]:

    def callback_on_step_end(pipeline: StableDiffusionPipeline, i: int, t: T, callback_kwargs: dict[str, T]) -> dict[str, T]:
        latents = callback_kwargs['latents']
        latents[0] = zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
        return {'latents': latents}
    return zts[offset], callback_on_step_end


@torch.no_grad()
def ddim_inversion(model: StableDiffusionPipeline, x0: np.ndarray, prompt: str, num_inference_steps: int, guidance_scale,) -> T:
    z0 = _encode_image(model, x0)
    model.scheduler.set_timesteps(num_inference_steps, device=z0.device)
    zs = _ddim_loop(model, z0, prompt, guidance_scale)
    return zs
