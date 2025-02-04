
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
import numpy as np
from PIL import Image
import os
import json
import random
import argparse
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T
from tqdm import tqdm
from models.edit_friendly_ddm.inversion_utils import inversion_forward_process, inversion_reverse_process
from models.edit_friendly_ddm.ptp_classes import AttentionReplace,AttentionRefine,AttentionStore
from models.edit_friendly_ddm.ptp_utils import register_attention_control
import torch.nn.functional as nnf
from torch.optim.adam import Adam
from torch import autocast, inference_mode

import pdb

from utils.utils import txt_draw,load_512,latent2image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

NUM_DDIM_STEPS = 50

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device,model_key):
        super().__init__()

        self.device = device
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", 
                                                 torch_dtype=torch.float32).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
                                                          torch_dtype=torch.float32).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
                                                         torch_dtype=torch.float32).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded stable diffusion!')


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent):
        latent_list=[latent]
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_list.append(latent)
        return latent_list

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        latent_list=[]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                    cond_batch = cond.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps
                    latent_list.append(x)
        return latent_list

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path,
                        inversion_prompt=''):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path)
        latent = self.encode_imgs(image)

        inverted_x = self.ddim_inversion(cond, latent)
        latent_reconstruction = self.ddim_sample(inverted_x[-1], cond)
        rgb_reconstruction = self.decode_latents(latent_reconstruction[-1])
        latent_reconstruction.reverse()
        return inverted_x, rgb_reconstruction, latent_reconstruction, 0


class Preprocess_NT(nn.Module):
    def __init__(self, device,model_key):
        super().__init__()

        self.device = device
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", 
                                                 torch_dtype=torch.float32).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
                                                          torch_dtype=torch.float32).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
                                                         torch_dtype=torch.float32).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded stable diffusion!')


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent):
        latent_list=[latent]
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_list.append(latent)
        return latent_list

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        latent_list=[]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(x.shape[0], 1, 1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (x - sigma * eps) / mu
                x = mu_prev * pred_x0 + sigma_prev * eps
                latent_list.append(x)
        return latent_list

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path,
                        inversion_prompt=''):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0) # torch.Size([1, 77, 768]), text embedding을 가져옴
        image = self.load_img(data_path)
        latent = self.encode_imgs(image).to(torch.float32)
        self.init_prompt(inversion_prompt)

        inverted_x = self.ddim_inversion(cond, latent) # torch.Size([1, 4, 64, 64]) * 50
        # null-text embedding을 optimization하는 부분 넣기
        with torch.enable_grad():
            uncond_embeddings = self.null_optimization(inverted_x, 10, 1e-5, 7.5)
        
        latent_reconstruction = self.ddim_sample(inverted_x[-1], cond)
        rgb_reconstruction = self.decode_latents(latent_reconstruction[-1])
        latent_reconstruction.reverse()
        # inverted_x : inversion을 통해 얻은 latent들
        # rgb_reconstruction : reconstrucion한 image
        # latent_reconstrucion : inversion을 통해 얻은 latent로 다시 forward 진행시켜서 얻은 latent들
        return inverted_x, rgb_reconstruction, latent_reconstruction, uncond_embeddings
    
    def null_optimization(self, latents, num_inner_steps, epsilon, guidance_scale):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)  # self.context : torch.Size([2, 77, 768])
        uncond_embeddings_list = []
        latent_cur = latents[-1]    # latent_cur : torch.Size([1, 4, 64, 64]), latents : torch.Size([1, 4, 64, 64]) * 51
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            t = self.scheduler.timesteps[i]
            if num_inner_steps!=0:
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                latent_prev = latents[len(latents) - i - 2]
                with torch.no_grad():
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)    # False
                for j in range(num_inner_steps):
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)    # False
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond) # False
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)    # False
                    loss = nnf.mse_loss(latents_prev_rec, latent_prev)  #False
                    optimizer.zero_grad()
                    loss.backward() 
                    optimizer.step()
                    loss_item = loss.item()
                    if loss_item < epsilon + i * 2e-5:
                        break

            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, guidance_scale, False, context)
        return uncond_embeddings_list
    
    def get_noise_pred_single(self, latents, t, context): # latents : torch.Size([1, 4, 64, 64]), context : torch.Size([1, 77, 768]) 둘 다 dytpe이 float32여야함.
        noise_pred = self.unet(latents, t, encoder_hidden_states=context)["sample"] # noise_pre : torch.Size([1, 4, 64, 64])
        return noise_pred
    
    def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents
    
    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.tokenizer(
            [""], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        self.context = torch.cat([uncond_embeddings.to(torch.float32), text_embeddings.to(torch.float32)])
        self.prompt = prompt
        
    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

ETA=1
SKIP=12

class Preprocess_EF(nn.Module):
    def __init__(self, device,model_key):
        super().__init__()

        self.device = device
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        # Create model
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(model_key).to(device)
        self.ldm_stable.scheduler = DDIMScheduler.from_config(model_key, subfolder = "scheduler")
        self.ldm_stable.scheduler.set_timesteps(NUM_DDIM_STEPS)
        print(f'[INFO] loaded stable diffusion!')


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent):
        latent_list=[latent]
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_list.append(latent)
        return latent_list

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        latent_list=[]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                    cond_batch = cond.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps
                    latent_list.append(x)
        return latent_list

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path, source_guidance_scale = 1, target_guidance_scale = 7.5, inversion_prompt=''):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path)
        latent = self.encode_imgs(image)    # w0

        wt, zs, wts = inversion_forward_process(self.ldm_stable, latent, etas=ETA, prompt=inversion_prompt, 
                                                cfg_scale=source_guidance_scale, prog_bar=True, num_inference_steps=NUM_DDIM_STEPS)
        # wt 1, zs 50, wts 51
        
        
        inverted_x = self.ddim_inversion(cond, latent)  # wts
        latent_reconstruction = self.ddim_sample(inverted_x[-1], cond)  # zs
        rgb_reconstruction = self.decode_latents(latent_reconstruction[-1])
        latent_reconstruction.reverse()
        
        zs.reverse()
        return wts, 0, zs, 0


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)


def register_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 3)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size:] = q[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]
                
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)

class PNP(nn.Module):
    def __init__(self, model_key,n_timesteps=NUM_DDIM_STEPS,device="cuda"):
        super().__init__()
        self.device = device

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float32).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(n_timesteps, device=self.device)
        self.n_timesteps=NUM_DDIM_STEPS
        print('SD model loaded')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self,image_path):
        # load image
        image = Image.open(image_path).convert('RGB') 
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        return image

    @torch.no_grad()
    def denoise_step(self, x, t,guidance_scale,noisy_latent):
        # register the time step and features in pnp injection modules
        latent_model_input = torch.cat(([noisy_latent]+[x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _,noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    @torch.no_grad()
    def denoise_step_NT(self, x, t,guidance_scale,noisy_latent, idx):
        # register the time step and features in pnp injection modules
        latent_model_input = torch.cat(([noisy_latent]+[x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds[idx], self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _,noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent
    
    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self,image_path,noisy_latent,target_prompt, uncond_embeddings, guidance_scale=7.5,pnp_f_t=0.8,pnp_attn_t=0.5):
        
        # load image
        self.image = self.get_data(image_path)
        self.eps = noisy_latent[-1]

        self.text_embeds = self.get_text_embeds(target_prompt, "ugly, blurry, black, low res, unrealistic")
        
        if uncond_embeddings is None:
            self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0] # size : torch.Size([1, 77, 768])
        else:
            self.pnp_guidance_embeds = uncond_embeddings 
        
        pnp_f_t = int(self.n_timesteps * pnp_f_t)
        pnp_attn_t = int(self.n_timesteps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        edited_img = self.sample_loop(self.eps,guidance_scale,noisy_latent, uncond_embeddings)
        
        return edited_img

    def sample_loop(self, x,guidance_scale,noisy_latent, uncond_embeddings):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                if uncond_embeddings is None:
                    x = self.denoise_step(x, t,guidance_scale,noisy_latent[-1-i])
                else:
                    x = self.denoise_step_NT(x, t,guidance_scale,noisy_latent[-1-i], i)
                    
            decoded_latent = self.decode_latent(x)
                
        return decoded_latent



image_save_paths={
    "ddim+pnp":"ddim+pnp",
    "null-text-inversion+pnp":"null-text-inversion+pnp",
    "directinversion+pnp":"directinversion+pnp",
    "edit-friendly-inversion+pnp":"edit-friendly-inversion+pnp"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method', type=str, default="ddim+pnp") # the editing methods that needed to run
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method=args.edit_method
    
    model_key = "CompVis/stable-diffusion-v1-4"
    toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    toy_scheduler.set_timesteps(NUM_DDIM_STEPS)

    timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=NUM_DDIM_STEPS,
                                                            strength=1.0,
                                                            device=device)
    if edit_method == "null-text-inversion+pnp":
        model = Preprocess_NT(device, model_key=model_key)
    elif edit_method == "edit-friendly-inversion+pnp":
        model = Preprocess_EF(device, model_key=model_key)
    else:
        model = Preprocess(device, model_key=model_key)
    pnp = PNP(model_key)

    def edit_image_ddim_PnP(
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512,512]
    ):
        torch.cuda.empty_cache()
        image_gt = load_512(image_path)
        _, rgb_reconstruction, latent_reconstruction, uncond_embeddings = model.extract_latents(data_path=image_path, 
                                            num_steps=NUM_DDIM_STEPS,
                                            inversion_prompt=prompt_src)
        # _은 size가 torch.Size([1, 4, 64, 64])인 tensor가 51개인 list, -> wts
        # rgb_reconstruction은 reconstruction된 이미지 size는 torch.Size([1, 3, 512, 512])
        # latent_reconstruction은 size가 torch.Size([1, 4, 64, 64])인 tensor가 50개인 list, -> zs
        edited_image=pnp.run_pnp(image_path,latent_reconstruction,prompt_tar,None,guidance_scale)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        return Image.fromarray(np.concatenate((
            image_instruct,
            image_gt,
            np.uint8(255*np.array(rgb_reconstruction[0].permute(1,2,0).cpu().detach())),
            np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
            ),1))



    def edit_image_directinversion_PnP(
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512,512]
    ):
        torch.cuda.empty_cache()
        image_gt = load_512(image_path)
        inverted_x, rgb_reconstruction, _, uncond_embeddings= model.extract_latents(data_path=image_path,
                                            num_steps=NUM_DDIM_STEPS,
                                            inversion_prompt=prompt_src)

        edited_image=pnp.run_pnp(image_path,inverted_x,prompt_tar,None, guidance_scale)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        return Image.fromarray(np.concatenate((
            image_instruct,
            image_gt,
            np.uint8(np.array(latent2image(model=pnp.vae, latents=inverted_x[1].to(pnp.vae.dtype))[0])),
            np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
            ),1))


    ## edit_image_nulltextinversion_PnP와 edit_image_EF_PnP method 추가

    def edit_image_null_text_inversion_PnP(
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512,512]
    ):
        torch.cuda.empty_cache()
        prompts = [prompt_src, prompt_tar]
        image_gt = load_512(image_path)
                                                                                
        wts, rgb_reconstruction, latent_reconstruction, uncond_embeddings = model.extract_latents(data_path=image_path, 
                                        num_steps=NUM_DDIM_STEPS,
                                        inversion_prompt=prompt_src)
        # _은 size가 torch.Size([1, 4, 64, 64])인 tensor가 51개인 list, -> wts
        # rgb_reconstruction은 reconstruction된 이미지 size는 torch.Size([1, 3, 512, 512])
        # latent_reconstruction은 size가 torch.Size([1, 4, 64, 64])인 tensor가 50개인 list, -> zs
        edited_image=pnp.run_pnp(image_path,latent_reconstruction,prompt_tar,uncond_embeddings,guidance_scale)
    
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        return Image.fromarray(np.concatenate((
            image_instruct,
            image_gt,
            np.uint8(255*np.array(rgb_reconstruction[0].permute(1,2,0).cpu().detach())),
            np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
            ),1))
            
    def edit_image_EF_PnP(
            image_path,
            prompt_src,
            prompt_tar,
            source_guidance_scale = 1,
            target_guidance_scale=7.5,
            image_shape=[512,512],
            pnp_f_t=0.8,
            pnp_attn_t=0.5
    ):
        # torch.cuda.empty_cache()
        # prompts = [prompt_src, prompt_tar]
        # image_gt = load_512(image_path)
        
        # _, rgb_reconstruction, latent_reconstruction, uncond_embeddings = model.extract_latents(data_path=image_path,
        #                                 num_steps=NUM_DDIM_STEPS,
        #                                 inversion_prompt=prompt_src)
        # edited_image=pnp.run_pnp(image_path,latent_reconstruction,prompt_tar,target_guidance_scale)
    
        # image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        # return Image.fromarray(np.concatenate((
        #     image_instruct,
        #     image_gt,
        #     np.uint8(255*np.array(rgb_reconstruction[0].permute(1,2,0).cpu().detach())),
        #     np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        #     ),1))
        torch.cuda.empty_cache()
        prompts = [prompt_src, prompt_tar]
        image_gt = load_512(image_path)
        image_gt = torch.from_numpy(image_gt).float() / 127.5 - 1
        image_gt = image_gt.permute(2, 0, 1).unsqueeze(0).to(device)
        with autocast("cuda"), inference_mode():
            w0 = (model.ldm_stable.vae.encode(image_gt).latent_dist.mode() * 0.18215).float()
            
        # controller = AttentionStore()
        # register_attention_control(model.ldm_stable, controller)
            
        wt, zs, wts = inversion_forward_process(model.ldm_stable, w0, etas=ETA, prompt=prompt_src, cfg_scale=source_guidance_scale, prog_bar=True, num_inference_steps=NUM_DDIM_STEPS)
        
        # controller = AttentionStore()
        # register_attention_control(model.ldm_stable, controller)
        
        x0_reconstruct, _ = inversion_reverse_process(model.ldm_stable, xT=wts[NUM_DDIM_STEPS-SKIP], etas=ETA, prompts=[prompt_tar], cfg_scales=[target_guidance_scale], prog_bar=True, zs=zs[:(NUM_DDIM_STEPS-SKIP)])

        cfg_scale_list = [source_guidance_scale, target_guidance_scale]
        prompts = [prompt_src, prompt_tar]
        # if (len(prompt_src.split(" ")) == len(prompt_tar.split(" "))):
        #     controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, model=model.ldm_stable)
        # else:
        #     # Should use Refine for target prompts with different number of tokens
        #     controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, model=model.ldm_stable)

        # register_attention_control(model.ldm_stable, controller)
        pnp_f_t = int(NUM_DDIM_STEPS * pnp_f_t)
        pnp_attn_t = int(NUM_DDIM_STEPS * pnp_attn_t)
        pnp.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        w0, _ = inversion_reverse_process(model.ldm_stable, xT=wts[NUM_DDIM_STEPS-SKIP], etas=ETA, prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[:(NUM_DDIM_STEPS-SKIP)])
        with autocast("cuda"), inference_mode():
            x0_dec = model.ldm_stable.vae.decode(1 / 0.18215 * w0[1].unsqueeze(0)).sample
            x0_reconstruct_edit = model.ldm_stable.vae.decode(1 / 0.18215 * w0[0].unsqueeze(0)).sample
            x0_reconstruct = model.ldm_stable.vae.decode(1 / 0.18215 * x0_reconstruct[0].unsqueeze(0)).sample
            
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
            
        return Image.fromarray(np.concatenate(
                                            (
                                                image_instruct,
                                                np.uint8((np.array(image_gt[0].permute(1,2,0).cpu().detach())/2+ 0.5)*255),
                                                np.uint8((np.array(x0_reconstruct_edit[0].permute(1,2,0).cpu().detach())/2+ 0.5)*255),
                                                np.uint8((np.array(x0_dec[0].permute(1,2,0).cpu().detach())/2+ 0.5)*255)
                                            ),
                                            1
                                            )
                            )
            


    def mask_decode(encoded_mask,image_shape=[512,512]):
        length=image_shape[0]*image_shape[1]
        mask_array=np.zeros((length,))
        
        for i in range(0,len(encoded_mask),2):
            splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
            for j in range(splice_len):
                mask_array[encoded_mask[i]+j]=1
                
        mask_array=mask_array.reshape(image_shape[0], image_shape[1])
        # to avoid annotation errors in boundary
        mask_array[0,:]=1
        mask_array[-1,:]=1
        mask_array[:,0]=1
        mask_array[:,-1]=1
                
        return mask_array

    
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    for key, item in editing_instruction.items():
        
        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
        if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
            print(f"editing image [{image_path}] with [{edit_method}]")
            setup_seed()
            torch.cuda.empty_cache()
            if edit_method=="ddim+pnp":
                edited_image = edit_image_ddim_PnP(
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=7.5,
                )
            elif edit_method=="directinversion+pnp":
                edited_image = edit_image_directinversion_PnP(
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=7.5,
                )
            elif edit_method=="null-text-inversion+pnp":
                edited_image = edit_image_null_text_inversion_PnP(
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=7.5,
                )
            elif edit_method=="edit-friendly-inversion+pnp":
                edited_image = edit_image_EF_PnP(
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    source_guidance_scale = 1,
                    target_guidance_scale=7.5,
                )
            else:
                raise NotImplementedError(f"No edit method named {edit_method}")
            
            
            if not os.path.exists(os.path.dirname(present_image_save_path)):
                os.makedirs(os.path.dirname(present_image_save_path))
            edited_image.save(present_image_save_path)
            
            print(f"finish")
            
        else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        
        