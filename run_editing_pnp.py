
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
from models.p2p.p2p_guidance_forward import p2p_guidance_forward
from models.pnp.preprocess import Preprocess, Preprocess_EF, Preprocess_NT
from models.pnp.pnp import PNP
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

ETA=1
SKIP=12

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
        # latent_reconstruction은 size가 torch.Size([2, 4, 64, 64])인 tensor가 50개인 list, -> zs
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
        
        
        