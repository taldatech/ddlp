"""
DiffuseDDLP modules.
Mostly adapted from lucidrains great diffusion library: https://github.com/lucidrains/denoising-diffusion-pytorch
"""

import datetime
import os
import math
from random import random
from functools import partial
from collections import namedtuple
from pathlib import Path
import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

from accelerate import Accelerator

from torch.utils.data import DataLoader

import cv2
import imageio
import matplotlib.pyplot as plt

# datasets
from datasets.get_dataset import get_video_dataset

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions

def animate_trajectories(orig_trajectory, path='./traj_anim.gif', duration=4 / 50, rec_to_pred_t=0):
    # rec_to_pred_t: the timestep from which prediction transitions from reconstruction to generation
    # prepare images
    font = cv2.FONT_HERSHEY_SIMPLEX
    origin = (5, 15)
    fontScale = 0.4
    color = (255, 255, 255)
    gt_border_color = (255, 0, 0)
    border_size = 2
    thickness = 1
    gt_traj_prep = []
    for i in range(orig_trajectory.shape[0]):
        image = (orig_trajectory[i] * 255).astype(np.uint8).copy()
        image = cv2.putText(image, f'GEN:{i}', origin, font, fontScale, color, thickness, cv2.LINE_AA)
        # add border
        image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                   value=gt_border_color)
        gt_traj_prep.append(image)

    total_images = []
    for i in range(len(orig_trajectory)):
        total_images.append(gt_traj_prep[i])
    imageio.mimsave(path, total_images, duration=duration)  # 1/50


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def clamp(value, min_value=None, max_value=None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


"""
EMA
"""


class EMA(nn.Module):
    """
    https://github.com/lucidrains/ema-pytorch/blob/main/ema_pytorch/ema_pytorch.py
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    def __init__(
            self,
            model,
            ema_model=None,
            # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
            beta=0.9999,
            update_after_step=100,
            update_every=10,
            inv_gamma=1.0,
            power=2 / 3,
            min_value=0.0,
            param_or_buffer_names_no_ema=set(),
            ignore_names=set(),
            ignore_startswith_names=set(),
            include_online_model=True
            # set this to False if you do not wish for the online model to be saved along with the ema model (managed externally)
    ):
        super().__init__()
        self.beta = beta

        # whether to include the online model within the module tree, so that state_dict also saves it

        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model]  # hack

        # ema model

        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = copy.deepcopy(model)
            except:
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        self.ema_model.requires_grad_(False)

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if
                                param.dtype in [torch.float, torch.float16]}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if
                             buffer.dtype in [torch.float, torch.float16]}

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema  # parameter or buffer

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0]))

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model),
                                                       self.get_params_iter(self.model)):
            ma_params.data.copy_(current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model),
                                                         self.get_buffers_iter(self.model)):
            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value=0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch <= 0:
            return 0.

        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model),
                                                          self.get_params_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            ma_params.data.lerp_(current_params.data, 1. - current_decay)

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model),
                                                          self.get_buffers_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            ma_buffer.data.lerp_(current_buffer.data, 1. - current_decay)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


"""
Particle Interaction Transformer (PINT)
"""


class SimpleRelativePositionalBias(nn.Module):
    # adapted from https://github.com/facebookresearch/mega
    def __init__(self, max_positions, num_heads=1, max_particles=None, layer_norm=False):
        super().__init__()
        self.max_positions = max_positions
        self.num_heads = num_heads
        self.max_particles = max_particles
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1, self.num_heads))
        self.ln_t = nn.LayerNorm([2 * max_positions - 1, self.num_heads]) if layer_norm else nn.Identity()

        if self.max_particles is not None:
            self.particle_rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_particles - 1, self.num_heads))
            self.ln_p = nn.LayerNorm([2 * max_particles - 1, self.num_heads]) if layer_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=std)
        if self.max_particles is not None:
            nn.init.normal_(self.particle_rel_pos_bias, mean=0.0, std=std)

    def get_particle_rel_position(self, num_particles):
        if self.max_particles is None:
            return 0.0
        if num_particles > self.max_particles:
            raise ValueError('Num particles {} going beyond max particles {}'.format(num_particles, self.max_particles))

        # seq_len * 2 -1
        in_ln = self.ln_p(self.particle_rel_pos_bias)
        b = in_ln[(self.max_particles - num_particles):(self.max_particles + num_particles - 1)]
        # print(f'b: {b}')
        # b = self.particle_rel_pos_bias[(self.max_particles - num_particles):(self.max_particles + num_particles - 1)]
        # seq_len * 3 - 1
        # t = F.pad(b, (0, seq_len))
        t = F.pad(b, (0, 0, 0, num_particles))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (num_particles, 1))
        t = t[:-num_particles]
        # seq_len x (3 * seq_len - 2)
        t = t.view(num_particles, 3 * num_particles - 2, b.shape[-1])
        r = (2 * num_particles - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        return t

    def forward(self, seq_len, num_particles=None):
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))

        # seq_len * 2 -1
        in_ln = self.ln_t(self.rel_pos_bias)
        b = in_ln[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, 0, 0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (seq_len, 1))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = t.view(seq_len, 3 * seq_len - 2, b.shape[-1])
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        p = None
        if num_particles is not None and self.max_particles is not None:
            p = self.get_particle_rel_position(num_particles)  # [1, n_heads, n_part, n_part]
            t = t[:, :, None, :, None, :]
            p = p[:, :, :, None, :, None]
        return t, p

    def extra_repr(self) -> str:
        return 'max positions={}'.format(self.max_positions)


class CausalParticleSelfAttention(nn.Module):
    """
    A particle-based multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1,
                 positional_bias=True, max_particles=None, linear_bias=False):
        super().__init__()
        assert n_embed % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # number of particles to consider in the self-attention
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, 1, block_size, 1, block_size))
        self.n_head = n_head
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        if self.positional_bias:
            self.rel_pos_bias = SimpleRelativePositionalBias(block_size, n_head, max_particles=max_particles)
        else:
            self.rel_pos_bias = nn.Identity()

    def forward(self, x):
        B, N, T, C = x.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        q = self.query(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)

        # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, N * T)
        att = att.view(B, -1, N, T, N, T)  # (B, nh, N, T, N, T)
        if self.positional_bias:
            if self.max_particles is not None:
                # print(f'num particles: {N}')
                bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
                bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
                att = att + bias_t + bias_p
            else:
                bias_t, _ = self.rel_pos_bias(T)
                bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                att = att + bias_t
        att = att.masked_fill(self.mask[:, :, :, :T, :, :T] == 0, float('-inf'))
        # print(f'att: {att.shape}')
        att = att.view(B, -1, N * T, N * T)  # (B, nh, N * T, N * T)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, N*T, N*T) x (B, nh, N*T, hs) -> (B, nh, N*T, hs)
        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)
        return y


class MLP(nn.Module):
    def __init__(self, n_embed, resid_pdrop=0.1, hidden_dim_multiplier=4, activation='gelu'):
        super().__init__()
        self.fc_1 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed)
        if activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(True)
        self.proj = nn.Linear(hidden_dim_multiplier * n_embed, n_embed)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = self.dropout(self.proj(self.act(self.fc_1(x))))
        return x


class PINTBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None):
        super().__init__()
        self.max_particles = max_particles
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = CausalParticleSelfAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                                positional_bias=positional_bias,
                                                max_particles=max_particles)
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ParticleTransformer(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, output_dim, attn_pdrop=0.1, resid_pdrop=0.1,
                 hidden_dim_multiplier=4, positional_bias=False,
                 activation='gelu', max_particles=None):
        super().__init__()
        self.positional_bias = positional_bias
        self.max_particles = max_particles  # for positional bias
        # input embedding stem
        if self.positional_bias:
            self.pos_emb = nn.Identity()
            # print(f'particle transformer: using relative positional bias')
        else:
            # print(f'particle transformer: using positional embeddings')
            self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embed))
        # transformer
        self.blocks = nn.Sequential(*[PINTBlock(n_embed, n_head, block_size, attn_pdrop,
                                                resid_pdrop, hidden_dim_multiplier,
                                                positional_bias, activation=activation,
                                                max_particles=max_particles)
                                      for _ in range(n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, output_dim, bias=False)

        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layer = n_layer

        # initialize layers
        self.apply(self._init_weights)
        if self.positional_bias:
            for m in self.blocks:
                m.attn.rel_pos_bias.reset_parameters()
        print(f"particle transformer # parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        std = 0.02
        # std = 0.05
        if isinstance(module, nn.Embedding):
            # pass
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            # pass
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ParticleTransformer):
            if not self.positional_bias:
                torch.nn.init.normal_(module.pos_emb, mean=0.0, std=std)

    def forward(self, x):
        b, n, t, f = x.size()
        # n is the number of particles
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        assert f == self.n_embed, "invalid particle feature dim"

        #         token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        if not self.positional_bias:
            position_embeddings = self.pos_emb[:, None, :t, :]
            #         x = self.drop(token_embeddings + position_embeddings)
            x = x + position_embeddings
        # x = self.blocks(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


class PINTDenoiser(nn.Module):
    def __init__(self, features_dim, hidden_dim, projection_dim,
                 n_head=4, n_layer=2, block_size=12, dropout=0.1,
                 predict_delta=True, positional_bias=True, max_particles=None, self_condition=False,
                 learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16):
        super(PINTDenoiser, self).__init__()
        """
        DLP Dynamics Module
        """
        self.features_dim = features_dim
        self.predict_delta = predict_delta
        self.projection_dim = projection_dim
        self.max_particles = max_particles  # for positional bias
        self.self_condition = self_condition
        # time embeddings

        time_dim = projection_dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(projection_dim)
            fourier_dim = projection_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, projection_dim)
        )

        self.particle_projection = nn.Sequential(nn.Linear(self.features_dim, hidden_dim),
                                                 # nn.ReLU(True),
                                                 nn.GELU(),
                                                 nn.Linear(hidden_dim, self.projection_dim))
        self.particle_transformer = ParticleTransformer(self.projection_dim, n_head, n_layer,
                                                        block_size, self.projection_dim,
                                                        attn_pdrop=dropout, resid_pdrop=dropout,
                                                        hidden_dim_multiplier=4,
                                                        positional_bias=positional_bias,
                                                        activation='gelu', max_particles=max_particles)
        self.particle_decoder = nn.Sequential(nn.Linear(self.projection_dim, hidden_dim),
                                              # nn.ReLU(True),
                                              nn.GELU(),
                                              nn.Linear(hidden_dim, self.features_dim))

    def forward(self, x, time, x_self_cond=None):
        # x: [bs, T, n_particles + 1, features_dim]
        # time: [bs, ]
        bs, timestep_horizon, n_particles, _ = x.shape
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        # time
        t = self.time_mlp(time)
        # project particles
        x_proj = self.particle_projection(x) + t[:, None, None, :]
        # [bs, T, n_particles + 1, projection_dim]

        x_proj = x_proj.permute(0, 2, 1, 3)
        # [bs, n_particles + 1, T, projection_dim]
        particles_trans = self.particle_transformer(x_proj)
        # [bs, n_particles + 1, T, projection_dim]
        particles_trans = particles_trans.permute(0, 2, 1, 3)
        # [bs, T, n_particles + 1, projection_dim]

        # decode transformer output
        particle_decoder_out = self.particle_decoder(particles_trans)
        # [bs, T, n_particles + 1, features_dim]
        return particle_decoder_out


"""
END Particle Interaction Transformer (PINT)
"""


class GaussianDiffusionPINT(nn.Module):
    def __init__(
            self,
            model,
            *,
            seq_length,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            objective='pred_noise',
            beta_schedule='cosine',
            p2_loss_weight_gamma=0.,
            p2_loss_weight_k=1,
            ddim_sampling_eta=0.,
            auto_normalize=False
    ):
        super().__init__()
        self.model = model
        self.n_particles = self.model.max_particles
        self.particle_dim = self.model.features_dim
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0',
                             'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond,
                                                                          clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
                                                                                 0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        seq_length, n_particles, p_dim = self.seq_length, self.n_particles, self.particle_dim
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, seq_length, n_particles, p_dim))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise=None, mask=None):
        # b, c, n = x_start.shape
        b, ts, n_particles, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction='none')
        if mask is not None:
            loss = loss * mask
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, mask=None, *args, **kwargs, ):
        # particles: [batch_size, t, n_particles + 1, particle_dim]
        b, ts, n_particles, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert ts == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, mask=mask, *args, **kwargs)


class TrainerDiffuseDDLP:
    def __init__(
            self,
            diffusion_model,
            ddlp_model,
            diffusion_config,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,  # 1000
            num_samples=25,
            results_folder='./results_diffpint',
            amp=False,
            fp16=False,
            split_batches=True,
            convert_image_to=None, seq_len=4, particle_norm=None, animation_fps=0.06
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
        )

        self.accelerator.native_amp = amp
        self.diffusion_config = diffusion_config
        self.model = diffusion_model
        self.ddlp_model = ddlp_model
        self.fg_features_dim = 2 + 2 + 2 + ddlp_model.learned_feature_dim
        # [xy, scale_xy, depth, obj_on, fg_features_dim]
        self.bg_features_dim = 2 + ddlp_model.bg_learned_feature_dim
        # [xy, bg_features_dim]
        self.n_particles = ddlp_model.n_kp_enc  # +1 for bg particle
        self.seq_len = seq_len
        self.particle_norm = particle_norm

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.animation_fps = animation_fps

        # dataset and dataloader
        ds = self.diffusion_config['ds']
        ds_root = self.diffusion_config['ds_root']
        seq_len = 50 if ds == 'traffic' else 100
        self.ds = get_video_dataset(ds, ds_root, mode='train', seq_len=seq_len)
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone=None):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': __version__
        }

        save_path = os.path.join(self.results_folder, 'saves', 'model.pth')
        torch.save(data, save_path)

    def load(self, milestone=None):
        accelerator = self.accelerator
        device = accelerator.device

        model_path = os.path.join(self.results_folder, 'saves', 'model.pth')
        if not os.path.exists(model_path):
            print(f'model checkpoint not found, training from scratch')
            return
        data = torch.load(model_path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")
        print(f'loaded model from checkpoint')

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def preprocess_ddlp_input(self, x):
        # x: [batch_size, timesteps, ch, im_size, im_size]
        batch_size, timesteps, ch, h, w = x.shape
        fg_dict = self.ddlp_model.fg_sequential_opt(x, deterministic=True, x_prior=x, reshape=True)
        # encoder
        z = fg_dict['z']
        z_features = fg_dict['z_features']
        z_obj_on = fg_dict['obj_on']
        z_depth = fg_dict['z_depth']
        z_scale = fg_dict['z_scale']

        # decoder
        bg_mask = fg_dict['bg_mask']

        x_in = x.view(-1, *x.shape[2:])  # [bs * T, ...]
        bg_dict = self.ddlp_model.bg_module(x_in, bg_mask, deterministic=True)
        z_bg = bg_dict['z_bg']
        z_kp_bg = bg_dict['z_kp']

        # collect and pad
        z_ddpm_fg = torch.cat([z, z_scale, z_depth, z_obj_on.unsqueeze(-1), z_features], dim=-1)
        # [batch_size * timesteps, n_kp, features]
        z_ddpm_fg = z_ddpm_fg.view(batch_size, -1, *z_ddpm_fg.shape[1:])
        # [batch_size, timesteps, n_kp, features]
        if self.particle_norm is not None:
            z_ddpm_fg, _ = self.particle_norm(z=z_ddpm_fg, normalize=True)
        fg_features_dim = z_ddpm_fg.shape[-1]
        # no padding needed here

        if self.particle_norm is not None:
            _, z_bg = self.particle_norm(z_bg=z_bg, normalize=True)
        z_ddpm_bg = torch.cat([z_kp_bg, z_bg.unsqueeze(1)], dim=-1)
        z_ddpm_bg = z_ddpm_bg.view(batch_size, -1, *z_ddpm_bg.shape[1:])
        bg_features_dim = z_ddpm_bg.shape[-1]
        # [batch_size, timesteps, 1, bg_features]
        # pad
        bg_padding = (0, fg_features_dim - z_ddpm_bg.shape[-1], 0, 0)
        z_ddpm_bg = F.pad(z_ddpm_bg, bg_padding)

        # cat
        z_ddpm = torch.cat([z_ddpm_fg, z_ddpm_bg], dim=-2)
        # [batch_size, timesteps, n_kp + 1, fg_features_dim]

        padding_mask = torch.ones_like(z_ddpm)
        padding_mask[:, :, z_ddpm_fg.shape[-2], bg_features_dim:] = 0.0
        return z_ddpm, padding_mask

    @torch.no_grad()
    def latent_to_ddlp_output(self, x):
        # x: [batch_size, timesteps. n_particles + 1, features_dim]
        batch_size, timesteps, total_particles, features_dim = x.shape
        # unpack
        z_ddpm_fg = x[:, :, :self.n_particles]
        if self.particle_norm is not None:
            z_ddpm_fg, _ = self.particle_norm(z=z_ddpm_fg, normalize=False)
        z = z_ddpm_fg[:, :, :, :2]
        z_scale = z_ddpm_fg[:, :, :, 2:4]
        z_depth = z_ddpm_fg[:, :, :, 4:5]
        z_obj_on = z_ddpm_fg[:, :, :, 5]
        z_features = z_ddpm_fg[:, :, :, 6:]

        z_ddpm_bg = x[:, :, self.n_particles, :self.bg_features_dim]
        z_bg_features = z_ddpm_bg[:, :, 2:]
        if self.particle_norm is not None:
            _, z_bg_features = self.particle_norm(z_bg=z_bg_features, normalize=False)

        z_dict = {'z': z, 'z_scale': z_scale, 'z_depth': z_depth, 'z_obj_on': z_obj_on,
                  'z_features': z_features, 'z_bg_features': z_bg_features}
        # decode
        z = z.reshape(-1, *z.shape[2:])
        z_features = z_features.reshape(-1, *z_features.shape[2:])
        z_bg_features = z_bg_features.reshape(-1, *z_bg_features.shape[2:])
        z_obj_on = z_obj_on.reshape(-1, *z_obj_on.shape[2:])
        z_depth = z_depth.reshape(-1, *z_depth.shape[2:])
        z_scale = z_scale.reshape(-1, *z_scale.shape[2:])
        dec_out = self.ddlp_model.decode_all(z, z_features, z_bg_features, z_obj_on,
                                             z_depth=z_depth, z_scale=z_scale)
        rec = dec_out['rec']
        rec = rec.reshape(batch_size, -1, *rec.shape[1:])
        return rec, z_dict

    @torch.no_grad()
    def unroll_z(self, z, z_scale, z_depth, z_obj_on, z_features, z_bg_features, cond_steps=10, horizon=50,
                 num_samples=1):
        # unroll dynamics
        z_v = z[:num_samples, :cond_steps]
        z_scale_v = z_scale[:num_samples, :cond_steps]
        z_depth_v = z_depth[:num_samples, :cond_steps]
        z_obj_on_v = z_obj_on[:num_samples, :cond_steps]
        z_features_v = z_features[:num_samples, :cond_steps]
        z_bg_features_v = z_bg_features[:num_samples, :cond_steps]
        batch_size, all_times = z_v.shape[0], z_v.shape[1]

        # dynamics
        dyn_out = self.ddlp_model.dyn_module.sample(z_v, z_scale_v, z_obj_on_v, z_depth_v, z_features_v,
                                                    z_bg_features_v,
                                                    steps=horizon - cond_steps, deterministic=True)
        z_dyn, z_scale_dyn, z_obj_on_dyn, z_depth_dyn, z_features_dyn, z_bg_features_dyn = dyn_out

        z_dyn = z_dyn.reshape(-1, *z_dyn.shape[2:])
        z_features_dyn = z_features_dyn.reshape(-1, *z_features_dyn.shape[2:])
        z_bg_features_dyn = z_bg_features_dyn.reshape(-1, *z_bg_features_dyn.shape[2:])
        z_obj_on_dyn = z_obj_on_dyn.reshape(-1, *z_obj_on_dyn.shape[2:])
        z_depth_dyn = z_depth_dyn.reshape(-1, *z_depth_dyn.shape[2:])
        z_scale_dyn = z_scale_dyn.reshape(-1, *z_scale_dyn.shape[2:])
        dec_out = self.ddlp_model.decode_all(z_dyn, z_features_dyn, z_bg_features_dyn, z_obj_on_dyn,
                                             z_depth=z_depth_dyn, z_scale=z_scale_dyn)
        rec_dyn = dec_out['rec']
        rec_dyn = rec_dyn.reshape(batch_size, -1, *rec_dyn.shape[1:])
        return rec_dyn

    @torch.no_grad()
    def unroll_z_dict(self, z_dict, cond_steps=10, horizon=50, num_samples=1):
        # unpack z_dict
        z = z_dict['z']
        z_scale = z_dict['z_scale']
        z_depth = z_dict['z_depth']
        z_obj_on = z_dict['z_obj_on']
        z_features = z_dict['z_features']
        z_bg_features = z_dict['z_bg_features']
        return self.unroll_z(z, z_scale, z_depth, z_obj_on, z_features, z_bg_features, cond_steps, horizon, num_samples)

    def sample(self, n_samples=1, fps=None):
        if fps is None:
            fps = self.animation_fps
        samples_save_path = os.path.join(self.results_folder, 'figures')
        os.makedirs(samples_save_path, exist_ok=True)
        with torch.no_grad():
            batches = num_to_groups(n_samples, self.batch_size)
            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
            all_images = torch.cat(all_images_list, dim=0)
            # [batch_size, timesteps, im_dim, im_dim]
            all_images, z_dict = self.latent_to_ddlp_output(all_images)
            rec = self.unroll_z_dict(z_dict, cond_steps=self.seq_len, horizon=100, num_samples=n_samples)
            # animate
            for i in range(rec.shape[0]):
                traj = rec[i].permute(0, 2, 3, 1).data.cpu().numpy()
                animate_trajectories(traj,
                                     path=os.path.join(samples_save_path, f'{i}_traj_anim.gif'),
                                     duration=fps)

    def sample_image(self, n_samples=1, times=(1, 10, 15, 20, 30, 40)):
        samples_save_path = os.path.join(self.results_folder, 'figures')
        os.makedirs(samples_save_path, exist_ok=True)
        with torch.no_grad():
            batches = num_to_groups(n_samples, self.batch_size)
            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
            all_images = torch.cat(all_images_list, dim=0)
            # [batch_size, timesteps, im_dim, im_dim]
            all_images, z_dict = self.latent_to_ddlp_output(all_images)
            rec = self.unroll_z_dict(z_dict, cond_steps=self.seq_len, horizon=100, num_samples=n_samples)
            for i in range(rec.shape[0]):
                traj = rec[i].permute(0, 2, 3, 1).data.cpu().numpy()
                curr_time = datetime.datetime.now().strftime("%d%m_%H%M")

                fig = plt.figure(figsize=(31, 15))
                title_size = 30
                plots_per_row = len(times)
                for j in range(len(times)):
                    ax = fig.add_subplot(1, plots_per_row, j + 1)
                    ax.imshow(traj[times[j]])
                    ax.set_axis_off()
                    ax.set_title(f't={times[j]}', fontsize=title_size)
                plt.tight_layout()
                save_path = os.path.join(samples_save_path, f'{curr_time}_{i}_diffuse_ddlp.png')
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        self.load()  # load checkpoint

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)[0][:, :self.seq_len].contiguous().to(device)
                    # pre-process data
                    x = data  # [batch_size, timesteps, ch, im_size, im_size]
                    # expected input shape: [batch_size,(n_kp + 1) * features_dim, timestesps]
                    data, padding_mask = self.preprocess_ddlp_input(x)

                    with self.accelerator.autocast():
                        loss = self.model(data, mask=padding_mask)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim=0)
                        # [batch_size, timesteps, im_dim, im_dim]
                        all_images, z_dict = self.latent_to_ddlp_output(all_images)
                        # [batch_size, timseteps, ch, im_size, im_size]
                        # all_images = all_images[:, :4].permute(1, 0, 2, 3, 4).reshape(-1, *all_images.shape[2:])
                        all_images = all_images[:, :4].reshape(-1, *all_images.shape[2:])
                        # print(f'all_images: {all_images.shape}')
                        fig_dir = os.path.join(self.results_folder, 'figures')
                        utils.save_image(all_images, os.path.join(fig_dir, f'sample-{milestone}.png'), nrow=4)
                        # make animation
                        if self.step % (5 * self.save_and_sample_every) == 0:
                            # unroll
                            rec = self.unroll_z_dict(z_dict, cond_steps=self.seq_len, horizon=100, num_samples=1)
                            # animate
                            traj = rec[0].permute(0, 2, 3, 1).data.cpu().numpy()
                            animate_trajectories(traj,
                                                 path=os.path.join(fig_dir,
                                                                   f'{milestone}_traj_anim.gif'),
                                                 duration=self.animation_fps)
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')


if __name__ == '__main__':
    batch_size = 32
    features_dim = 2 + 2 + 2 + 3
    hidden_dim = 256
    projection_dim = 256
    n_particles = 10
    seq_len = 4
    model = PINTDenoiser(features_dim, hidden_dim, projection_dim,
                         n_head=8, n_layer=6, block_size=4, dropout=0.1,
                         predict_delta=False, positional_bias=True, max_particles=n_particles + 1, self_condition=False,
                         learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16)

    in_particles = torch.randn(batch_size, seq_len, n_particles + 1, features_dim)
    t = torch.randint(0, 1000, (batch_size,), device=in_particles.device).long()
    model_out = model(in_particles, t)
    print(f'model_out: {model_out.shape}')

    diffusion = GaussianDiffusionPINT(
        model,
        seq_length=seq_len,
        timesteps=1000,  # number of steps
        loss_type='l1',  # L1 or L2
        objective='pred_x0',
    )
    loss = diffusion(in_particles)
    # after a lot of training

    sampled_images = diffusion.sample(batch_size=4)
    print(sampled_images.shape)  # (4, 3, 128, 128)
