from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionXLPipeline
from models.stylealign.inversion import NegativePromptInversion, NullInversion, DirectInversion
from models.stylealign import inversion_sd, inversion_sdxl
from utils.utils import load_512, latent2image, txt_draw
import torch
import mediapy
from models.stylealign import sa_handler
import math
import numpy as np
from PIL import Image

import pdb


class StyleAlign:
    def __init__(self, method_list, device, num_ddim_steps=50) -> None:
        self.device=device
        self.method_list=method_list
        self.num_ddim_steps=num_ddim_steps
        #   init model
        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        clip_sample=False, set_alpha_to_one=False)
        self.ldm_stable = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, variant="fp16",
                                        use_safetensors=True,
                                        scheduler=self.scheduler).to(device)
        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)

        
    def __call__(self, 
                edit_method,
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                proximal=None,
                quantile=0.7,
                use_reconstruction_guidance=False,
                recon_t=400,
                recon_lr=0.1,
                cross_replace_steps=0.4,
                self_replace_steps=0.6,
                blend_word=None,
                eq_params=None,
                is_replace_controller=False,
                use_inversion_guidance=False,
                dilate_mask=1,):
        if edit_method=="ddim+stylealign":
            return self.edit_image_ddim(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        
        elif edit_method in "null-text-inversion+stylealign":
            return self.edit_image_null_text_inversion(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        
        elif edit_method=="negative-prompt-inversion+stylealign":
            return self.edit_image_negative_prompt_inversion(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar,
                                        guidance_scale=guidance_scale, proximal=None, quantile=quantile, use_reconstruction_guidance=use_reconstruction_guidance,
                                        recon_t=recon_t, recon_lr=recon_lr, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, blend_word=blend_word, eq_params=eq_params,
                                        is_replace_controller=is_replace_controller, use_inversion_guidance=use_inversion_guidance,
                                        dilate_mask=dilate_mask)
        elif edit_method=="directinversion+stylealign":
            return self.edit_image_directinversion(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method=='edit-friendly-inversion+stylealign':
            return self.edit_image_EF(edit_method=edit_method, image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, source_guidance_scale=1,
                        target_guidance_scale=7.5, cross_replace_steps=0.4, self_replace_steps=0.6)
        else:
            raise NotImplementedError(f"No edit method named {edit_method}")
    def edit_image_ddim(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=10,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        zts = inversion_sdxl.ddim_inversion(self.ldm_stable, image_gt, prompt_src, self.num_ddim_steps, 2)
        
        # some parameters you can adjust to control fidelity to reference
        shared_score_shift = np.log(2)  # higher value induces higher fidelity, set 0 for no shift
        shared_score_scale = 1.0  # higher value induces higher, set 1 for no rescale

        handler = sa_handler.Handler(self.ldm_stable)
        sa_args = sa_handler.StyleAlignedArgs(
            share_group_norm=True, share_layer_norm=True, share_attention=True,
            adain_queries=True, adain_keys=True, adain_values=False,
            shared_score_shift=shared_score_shift, shared_score_scale=shared_score_scale,)
        handler.register(sa_args)
        
        zT, inversion_callback = inversion_sd.make_inversion_callback(zts, offset=5)
        
        g_cpu = torch.Generator(device='cpu')
        g_cpu.manual_seed(1234)
        
        latents = torch.randn(len(prompts), 4, 64, 64, device='cpu', generator=g_cpu,
                      dtype=self.ldm_stable.unet.dtype,).to('cuda:0')
        
        latents[0] = zT
        
        images = self.ldm_stable(prompts, latents=latents,
                    callback_on_step_end=inversion_callback,
                    num_inference_steps=self.num_ddim_steps, guidance_scale=guidance_scale).images

        handler.remove()
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, np.array(images[0]),np.array(images[-1])),axis=1))
        
        # for very famouse images consider supressing attention to refference, here is a configuration example:
        # shared_score_shift = np.log(1)
        # shared_score_scale = 0.5

        # controller = AttentionStore()
        # reconstruct_latent, x_t = p2p_guidance_forward(model=self.ldm_stable, 
        #                                prompt=[prompt_src], 
        #                                controller=controller, 
        #                                latent=x_t, 
        #                                num_inference_steps=self.num_ddim_steps, 
        #                                guidance_scale=guidance_scale, 
        #                                generator=None, 
        #                                uncond_embeddings=uncond_embeddings)
        

        # reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        # image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        # ########## edit ##########
        # cross_replace_steps = {
        #     'default_': cross_replace_steps,
        # }

        # controller = make_controller(pipeline=self.ldm_stable,
        #                             prompts=prompts,
        #                             is_replace_controller=is_replace_controller,
        #                             cross_replace_steps=cross_replace_steps,
        #                             self_replace_steps=self_replace_steps,
        #                             blend_words=blend_word,
        #                             equilizer_params=eq_params,
        #                             num_ddim_steps=self.num_ddim_steps,
        #                             device=self.device)
        # latents, _ = p2p_guidance_forward(model=self.ldm_stable, 
        #                                prompt=prompts, 
        #                                controller=controller, 
        #                                latent=x_t, 
        #                                num_inference_steps=self.num_ddim_steps, 
        #                                guidance_scale=guidance_scale, 
        #                                generator=None, 
        #                                uncond_embeddings=uncond_embeddings)

        # images = latent2image(model=self.ldm_stable.vae, latents=latents)

        # return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_null_text_inversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NullInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, uncond_embeddings = null_inversion.invert(               # ddim latents, optimized uncond_embeddings를 얻는다.
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=[prompt_src], 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)
        

        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        latents, _ = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    
    def edit_image_negative_prompt_inversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        proximal=None,
        quantile=0.7,
        use_reconstruction_guidance=False,
        recon_t=400,
        recon_lr=0.1,
        npi_interp=0,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        use_inversion_guidance=False,
        dilate_mask=1,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NegativePromptInversion(model=self.ldm_stable,
                                                num_ddim_steps=self.num_ddim_steps)
        _, image_enc_latent, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src, npi_interp=npi_interp)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = proximal_guidance_forward(
                    model=self.ldm_stable,
                    prompt=[prompt_src],
                    controller=controller,
                    latent=x_t,
                    guidance_scale=guidance_scale,
                    generator=None,
                    uncond_embeddings=uncond_embeddings,
                    edit_stage=False,
                    prox=None,
                    quantile=quantile,
                    image_enc=None,
                    recon_lr=recon_lr,
                    recon_t=recon_t,
                    inversion_guidance=False,
                    x_stars=None,
                    dilate_mask=dilate_mask)
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        
        latents, _ = proximal_guidance_forward(
                        model=self.ldm_stable,
                        prompt=prompts,
                        controller=controller,
                        latent=x_t,
                        guidance_scale=guidance_scale,
                        generator=None,
                        uncond_embeddings=uncond_embeddings,
                        edit_stage=True,
                        prox=proximal,
                        quantile=quantile,
                        image_enc=image_enc_latent if use_reconstruction_guidance else None,
                        recon_lr=recon_lr
                            if use_reconstruction_guidance or use_inversion_guidance else 0,
                        recon_t=recon_t
                            if use_reconstruction_guidance or use_inversion_guidance else 1000,
                        x_stars=x_stars,
                        dilate_mask=dilate_mask)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)


        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_directinversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale)    # ddim latent들과 차이 list를 구함
        x_t = x_stars[-1]   # z_0

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]  # reconstruct_latent : z^*_0

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        # editing된 latent만들기 
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents) 

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    
    def edit_image_EF(
        self,
        edit_method,
        image_path,
        prompt_src,
        prompt_tar,
        source_guidance_scale=1,
        target_guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6
        ):
        ETA=1
        SKIP=12
        image_gt = load_512(image_path)
        
        image_gt = torch.from_numpy(image_gt).float() / 127.5 - 1
        image_gt = image_gt.permute(2, 0, 1).unsqueeze(0).to(self.device)
        with autocast("cuda"), inference_mode():
            w0 = (self.ldm_stable.vae.encode(image_gt).latent_dist.mode() * 0.18215).float()
            
        controller = AttentionStore()
        register_attention_control(self.ldm_stable, controller)
            
        wt, zs, wts = inversion_forward_process(self.ldm_stable, w0, etas=ETA, prompt=prompt_src, cfg_scale=source_guidance_scale, prog_bar=True, num_inference_steps=self.num_ddim_steps)
        
        controller = AttentionStore()
        register_attention_control(self.ldm_stable, controller)
        
        x0_reconstruct, _ = inversion_reverse_process(self.ldm_stable, xT=wts[self.num_ddim_steps-SKIP], etas=ETA, prompts=[prompt_tar], cfg_scales=[target_guidance_scale], prog_bar=True, zs=zs[:(self.num_ddim_steps-SKIP)], controller=controller)

        cfg_scale_list = [source_guidance_scale, target_guidance_scale]
        prompts = [prompt_src, prompt_tar]
        if (len(prompt_src.split(" ")) == len(prompt_tar.split(" "))):
            controller = AttentionReplace(prompts, self.num_ddim_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, model=self.ldm_stable)
        else:
            # Should use Refine for target prompts with different number of tokens
            controller = AttentionRefine(prompts, self.num_ddim_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, model=self.ldm_stable)

        register_attention_control(self.ldm_stable, controller)
        w0, _ = inversion_reverse_process(self.ldm_stable, xT=wts[self.num_ddim_steps-SKIP], etas=ETA, prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[:(self.num_ddim_steps-SKIP)], controller=controller)
        with autocast("cuda"), inference_mode():
            x0_dec = self.ldm_stable.vae.decode(1 / 0.18215 * w0[1].unsqueeze(0)).sample
            x0_reconstruct_edit = self.ldm_stable.vae.decode(1 / 0.18215 * w0[0].unsqueeze(0)).sample
            x0_reconstruct = self.ldm_stable.vae.decode(1 / 0.18215 * x0_reconstruct[0].unsqueeze(0)).sample
            
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
