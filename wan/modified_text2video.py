# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm
from torchvision.io import read_video
from torchvision.io import write_video
from torchvision.transforms import ToPILImage
from torch.cuda.amp import GradScaler
from PIL import Image
import numpy as np
from diffusers.video_processor import VideoProcessor

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .guidance_utils.motion_flow_utils import compute_motion_flow

def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        self.video_processor = VideoProcessor()

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt


    @torch.no_grad()
    def load_latent(self):
        """ Load video and pass through VAE encoder"""
        data_path = self.config.video_path
        
        def save_video(video, path):
            video_codec = "libx264"
            video_options = {
                "crf": "17",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
                "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
            }   
            write_video(
                path,
                video,
                fps=10,
                video_codec=video_codec,
                options=video_options,
            )
        if data_path.endswith(".mp4"):
            video = read_video(data_path, pts_unit='sec')[0].permute(0, 3, 1, 2).cuda() / 255
            video = [ToPILImage()(video[i]).resize(self.resolution) for i in range(video.shape[0])]
        else:
            images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
            images = sorted(images, key=lambda x: int(x.stem.split('f')[-1]))
            video = [Image.open(img).resize(self.resolution).convert('RGB') for img in images]
            
        video = video[: self.config.video_length]
        save_video([np.array(img) for img in video], str(Path(self.config.output_path) / f"original.mp4"))

        video = self.video_processor(video)
        video = video.to(self.vae.dtype).to("cuda")
        latents = self.vae.encode([video])[0]  
        
        latents = latents.permute(0,2,1,3,4)
        
        return latents # Will it works?
    
    @torch.no_grad()
    def load_features(self, moft=False):
        """Load saved features for motion video
        moft: Whether to compute motion channels for MOFT method"""
        
        motion_features = {}
        motion_channels = {}
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        # What is the size?

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        arg = {'context': self.config["source_prompt"], 'seq_len': seq_len} # Source Prompt not added yet
        with torch.autocast(device_type="cuda"):
            self.model(
                x=self.motion_latent.to('cuda'),
                t=self.motion_timestep,
                **arg
            )
            
        for block_id in self.config.guidance_blocks:
            module = self.model.blocks[block_id] # config not modified
            orig_features = module.saved_features # Not instiantiaed using modulewithguidance yet
            motion_features[module.block_name] = orig_features
            
            if moft:
                orig_norm = orig_features - torch.mean(orig_features, axis=0)[None]
                num_frames, c, h, w = orig_norm.shape
                channels = orig_norm.permute(0,2,3,1).reshape(-1, c)
                _, _, Vt = torch.linalg.svd(channels.to(torch.float32), full_matrices=False)
                top_n = list(torch.argsort(torch.abs(Vt[0]), descending=True)[:int(self.config.prop_motion*c)])
                motion_channels[module.block_name] = top_n
        if moft:
            return motion_features, motion_channels
        return motion_features
    
    @torch.no_grad()
    def load_attn_features(self):
        """ 🔍 AMF Extraction """
        for block_id in self.config.guidance_blocks:
            self.model.blocks[block_id].self_attn.inject_kv = False
            self.model.blocks[block_id].self_attn.copy_kv = True
            # The self attention need to be created with a block name!
            
        attn_features = {}
        # Store keys and queries for all attention blocks
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        # What is the size?

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        arg = {'context': self.config["source_prompt"], 'seq_len': seq_len} # Source Prompt not added yet
        with torch.autocast(device_type="cuda"):
            self.model(
                x=self.motion_latent.to('cuda'),
                t=self.motion_timestep,
                **arg
            )
            
        for block_id in self.config.guidance_blocks:
            module = self.model.blocks[block_id].self_attn # config not modified
            attn_features[module.block_name] = compute_motion_flow(module.query, module.key, 
                                                    h=self.patches_height, 
                                                    w=self.patches_width, 
                                                    temp=self.config.motion_temp, 
                                                    argmax=self.config.argmax_motion_flow)
            self.model.blocks[block_id].self_attn.copy_kv = False
            self.model.blocks[block_id].self_attn.key = None
            self.model.blocks[block_id].self_attn.value = None  
            self.model.blocks[block_id].self_attn.query = None
            
        return attn_features
    
    def change_mode(self, train=True):
        pass
                
    ############################## GUIDANCE LOSS FUNCTIONS ##############################
    def compute_motion_flow_loss(self, x, ts, rope=None, pos_emb=None):
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        # What is the size?

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        arg = {'context': self.config["source_prompt"], 'seq_len': seq_len} # Source Prompt not added yet
        with torch.autocast(device_type="cuda"):
            self.model(
                x=self.motion_latent.to('cuda'),
                t=self.motion_timestep,
                **arg
            )
            
        # Attention guidance
        total_loss = 0
        for block_id in self.config.guidance_blocks:
            module = self.model.blocks[block_id].self_attn # config not modified
            motion_flow = compute_motion_flow(module.query, module.key,
                                                    h=self.patches_height, 
                                                    w=self.patches_width, 
                                                    temp=self.config.motion_temp, 
                                                    argmax=self.config.argmax_motion_flow)
            ref_motion_flow = self.motion_attn_features[module.block_name].detach()
            
            # Threshold loss on motion flow (d x 1350 x 2) for d displacement maps
            if self.config.threshloss:
                flow_norms = torch.norm(ref_motion_flow, dim=-1)
                idxs = flow_norms > 0
                attn_loss = F.mse_loss(ref_motion_flow[idxs], motion_flow[idxs])
            else:
                attn_loss = F.mse_loss(ref_motion_flow, motion_flow)

            total_loss += attn_loss
        if len(self.config.guidance_blocks) > 0:
            total_loss /= len(self.config.guidance_blocks)
        
        for block_id in self.config.guidance_blocks:
            self.model.blocks[block_id].self_attn.query = None
            self.model.blocks[block_id].self_attn.key = None
            self.model.blocks[block_id].self_attn.value = None
        return total_loss
    
    def compute_moft_loss(self, x, ts, rope=None, pos_emb=None):
        """Motion Feature (MOFT) Loss"""
        def compute_MOFT(orig, 
                 target, 
                 motion_channels,
                 ):
            # Compute motion channels from current video only (T x C x H x W) and extract top prop_motion% channels
            orig_norm = orig - torch.mean(orig, axis=0)[None]
            target_norm = target - torch.mean(target, axis=0)[None]

            features_diff_loss = 0
            for f in range(orig_norm.shape[0]):
                top_n = motion_channels
                orig_moft_f = orig_norm[f, top_n]
                target_moft_f = target_norm[f, top_n]
                features_diff_loss += 1 - F.cosine_similarity(target_moft_f, orig_moft_f.detach(), dim=0).mean()

            features_diff_loss /= orig_norm.shape[0]
            return features_diff_loss
        
        target_features = {}
        
        
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        # What is the size?

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        arg = {'context': self.config["source_prompt"], 'seq_len': seq_len} # Source Prompt not added yet
        with torch.autocast(device_type="cuda"):
            self.model(
                x=self.motion_latent.to('cuda'),
                t=self.motion_timestep,
                **arg
            )
            
            
        total_loss = 0
        for block_id in self.config.guidance_blocks:
            module = self.model.blocks[block_id].self_attn
            target_name = module.block_name
            target_features = module.saved_features

            orig_features = self.motion_orig_features[target_name]
            motion_channels = self.motion_channels[target_name]

            loss = compute_MOFT(
                orig_features.detach(), 
                target_features,
                motion_channels,
            )
            total_loss += loss
        if len(self.config.guidance_blocks) > 0:
            total_loss /= len(self.config.guidance_blocks)
        
        return total_loss

    def compute_smm_loss(self, x, ts, rope=None, pos_emb=None):
        """Spatial Marginal Mean Loss"""
        def compute_SMM(orig, 
                        target,
                        ):
            # Take spatial mean
            orig_smm = orig.mean(dim=(-1, -2), keepdim=True)
            target_smm = target.mean(dim=(-1, -2), keepdim=True)

            features_diff_loss = 0
            for f in range(orig_smm.shape[0]):
                orig_anchor = orig_smm[f]
                target_anchor = orig_smm[f]
                orig_diffs = orig_smm - orig_anchor  # t d 1 1
                target_diffs = target_smm - target_anchor  # t d 1 1
                features_diff_loss += 1 - F.cosine_similarity(target_diffs, orig_diffs.detach(), dim=0).mean()
            features_diff_loss /= orig_smm.shape[0]
            return features_diff_loss
    
        target_features = {}
        
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        # What is the size?

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        arg = {'context': self.config["source_prompt"], 'seq_len': seq_len} # Source Prompt not added yet
        with torch.autocast(device_type="cuda"):
            self.model(
                x=self.motion_latent.to('cuda'),
                t=self.motion_timestep,
                **arg
            )
            
        total_loss = 0
        for block_id in self.config.guidance_blocks:
            module = self.model.blocks[block_id].self_attn
            target_name = module.block_name
            target_features = module.saved_features

            orig_features = self.motion_orig_features[target_name]

            loss = compute_SMM(
                orig_features.detach(), 
                target_features,
            )
            total_loss += loss
        if len(self.config.guidance_blocks) > 0:
            total_loss /= len(self.config.guidance_blocks)
        
        return total_loss
        

    ############################## GUIDED DENOISING METHODS ##############################
    def guidance_step(self, x, i, t, mode, loss_type):
        """⚙️ Motion Optimization
        Optimisation at single denoising step for number of steps
        mode: rope, posemb, latent (object being optimized)
        loss_type: flow, moft, smm (loss computation)
        """
        for block_id in self.config.guidance_blocks:
            self.transformer.transformer_blocks[block_id].self_attn.inject_kv = False
            self.transformer.transformer_blocks[block_id].self_attn.copy_kv = True
        
        lr = self.lr_range[i]
        optimized_emb = None
        optimized_rope = None
        self.change_mode(train=True)
        
        scaler = GradScaler()

        if loss_type == "flow":
            loss_method = self.compute_motion_flow_loss
        elif loss_type == "moft":
            loss_method = self.compute_moft_loss
        elif loss_type == "smm":
            loss_method = self.compute_smm_loss
        else:
            print("Invalid loss type")
        
        if mode=="rope":
            if self.transformer.trainable_rope is None:
                optimized_rope = torch.stack([self.transformer.init_rope, self.transformer.init_rope], dim=0)
            else:
                optimized_rope = self.transformer.trainable_rope
            
            optimized_rope = optimized_rope.clone().detach().to(dtype=torch.float32, device=self.device).requires_grad_(True)
            optimizer = torch.optim.Adam([optimized_rope], lr=lr)

            for step_i in tqdm(range(self.config.optimization_steps)):
                optimizer.zero_grad()

                total_loss = loss_method(x, t, rope=optimized_rope)
                
                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                scaler.scale(total_loss).backward()

                scaler.step(optimizer)
                scaler.update()
                clean_memory()
            
            self.transformer.trainable_rope = optimized_rope.detach() # Not implemented
            if self.config.save_embeds:
                os.makedirs(os.path.join(self.output_path, 'embeds'), exist_ok=True)
                torch.save(optimized_rope.detach(), os.path.join(self.output_path, 'embeds', f"rope_{t}.pt"))
            optimized_x = x
        elif mode == "posemb":
            if self.transformer.trainable_pos_embedding is None:
                text_seq_length = self.config.text_seq_length
                seq_length = self.patches_height * self.patches_width * self.latent_num_frames
                optimized_emb = self.transformer.init_pos_embedding[:, text_seq_length:(text_seq_length+seq_length)].clone().detach().to(dtype=torch.float32, device=self.device).requires_grad_(True)
            else:
                optimized_emb = self.transformer.trainable_pos_embedding.clone().detach().to(dtype=torch.float32, device=self.device).requires_grad_(True)

            optimizer = torch.optim.Adam([optimized_emb], lr=lr)

            for step_i in tqdm(range(self.config.optimization_steps)):
                optimizer.zero_grad()

                total_loss = loss_method(x, t, pos_emb=optimized_emb)

                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                scaler.scale(total_loss).backward()

                scaler.step(optimizer)
                scaler.update()
                clean_memory()
            self.transformer.trainable_pos_embedding = optimized_emb.detach() # Not implemented
            if self.config.save_embeds:
                os.makedirs(os.path.join(self.output_path, 'embeds'), exist_ok=True)
                torch.save(optimized_emb.detach(), os.path.join(self.output_path, 'embeds', f"posemb_{t}.pt"))
            optimized_x = x
        elif mode=="latent":
            optimized_x = x.clone().detach().to(dtype=torch.float32).requires_grad_(True)
            optimizer = torch.optim.Adam([optimized_x], lr=lr)

            for step_i in tqdm(range(self.config.optimization_steps)):
                optimizer.zero_grad()

                total_loss = loss_method(optimized_x, t)
                
                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                scaler.scale(total_loss).backward()

                scaler.step(optimizer)
                scaler.update()
            
            if self.config.save_embeds:
                os.makedirs(os.path.join(self.output_path, 'embeds'), exist_ok=True)
                torch.save(optimized_x, os.path.join(self.output_path, 'embeds', f"latent_{t}.pt"))
                
        self.change_mode(train=False)
        return optimized_x.detach(), optimized_emb, optimized_rope

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
