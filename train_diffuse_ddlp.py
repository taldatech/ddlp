"""
Main training function of DiffuseDDLP
"""

import argparse
import os
import shutil
import json
from utils.util_func import prepare_logdir, get_config
from tqdm.auto import tqdm

# torch
import torch
from torch.utils.data import DataLoader

# datasets
from datasets.get_dataset import get_video_dataset

# models
from modules.diffusion_modules import TrainerDiffuseDDLP, GaussianDiffusionPINT, PINTDenoiser
from models import ObjectDynamicsDLP

"""
Particle Normalization
Calculate and save the latent statistics of particles for normalization/standardization purposes.
Denoisers' input is usually normalized, thus, we need to calculate the statistics of the particles.
"""


class ParticleNormalization(torch.nn.Module):
    def __init__(self, config, mode='minmax', eps=1e-5):
        super().__init__()
        assert mode in ["minmax", "std"], f'mode: {mode} not supported'
        self.diffusion_config = config
        self.root = config['ddlp_dir']
        self.eps = eps
        self.ds = config['ds']
        device = config['device']
        if 'cuda' in device:
            device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        self.device = device
        self.mode = mode
        self.ddlp_dir = config['ddlp_dir']
        self.ddlp_ckpt = config['ddlp_ckpt']
        ddlp_conf = os.path.join(self.ddlp_dir, 'hparams.json')
        ddlp_config = get_config(ddlp_conf)
        self.config = ddlp_config
        self.particle_feature_dim = self.config['learned_feature_dim']
        self.fg_total_dim = 2 + 2 + 2 + self.particle_feature_dim  # (x, y), (scale_x, scale_y), depth, transparency
        self.bg_total_dim = self.particle_feature_dim
        mu = torch.zeros(self.fg_total_dim)
        self.register_buffer('mu', mu)
        mu_bg = torch.zeros(self.bg_total_dim)
        self.register_buffer('mu_bg', mu_bg)
        std = torch.ones(self.fg_total_dim)
        self.register_buffer('std', std)
        std_bg = torch.ones(self.bg_total_dim)
        self.register_buffer('std_bg', std_bg)
        min_val = torch.zeros(self.fg_total_dim)
        self.register_buffer('min_val', min_val)
        max_val = torch.zeros(self.fg_total_dim)
        self.register_buffer('max_val', max_val)
        min_val_bg = torch.zeros(self.bg_total_dim)
        self.register_buffer('min_val_bg', min_val_bg)
        max_val_bg = torch.zeros(self.bg_total_dim)
        self.register_buffer('max_val_bg', max_val_bg)
        # get statistics
        self.get_latent_statistics()
        print(f'mu: {self.mu}, std: {self.std}, min: {self.min_val}, max: {self.max_val}')

    def get_latent_statistics(self):
        stats_path = os.path.join(self.root, 'latent_stats.pth')
        if os.path.exists(stats_path):
            params = torch.load(stats_path)
            self.load_state_dict(params)
            print(f'latent stats loaded from {stats_path}')
        else:
            # calculate stats
            print(f'latent stats not found, calculating stats...')
            self.calc_latent_stats()

    def calc_latent_stats(self, ):
        # load model
        ddlp_config = self.config
        ddlp_ckpt = self.ddlp_ckpt
        device = self.device
        # load model
        image_size = ddlp_config['image_size']
        ch = ddlp_config['ch']
        enc_channels = ddlp_config['enc_channels']
        prior_channels = ddlp_config['prior_channels']
        use_correlation_heatmaps = ddlp_config['use_correlation_heatmaps']
        enable_enc_attn = ddlp_config['enable_enc_attn']
        filtering_heuristic = ddlp_config['filtering_heuristic']

        model = ObjectDynamicsDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                                  image_size=image_size, n_kp=ddlp_config['n_kp'],
                                  learned_feature_dim=ddlp_config['learned_feature_dim'],
                                  pad_mode=ddlp_config['pad_mode'],
                                  sigma=ddlp_config['sigma'],
                                  dropout=ddlp_config['dropout'], patch_size=ddlp_config['patch_size'],
                                  n_kp_enc=ddlp_config['n_kp_enc'],
                                  n_kp_prior=ddlp_config['n_kp_prior'], kp_range=ddlp_config['kp_range'],
                                  kp_activation=ddlp_config['kp_activation'],
                                  anchor_s=ddlp_config['anchor_s'],
                                  use_resblock=ddlp_config['use_resblock'],
                                  timestep_horizon=ddlp_config['timestep_horizon'],
                                  predict_delta=ddlp_config['predict_delta'],
                                  scale_std=ddlp_config['scale_std'],
                                  offset_std=ddlp_config['offset_std'], obj_on_alpha=ddlp_config['obj_on_alpha'],
                                  obj_on_beta=ddlp_config['obj_on_beta'], pint_heads=ddlp_config['pint_heads'],
                                  pint_layers=ddlp_config['pint_layers'], pint_dim=ddlp_config['pint_dim'],
                                  use_correlation_heatmaps=use_correlation_heatmaps,
                                  enable_enc_attn=enable_enc_attn, filtering_heuristic=filtering_heuristic).to(device)
        model.load_state_dict(torch.load(ddlp_ckpt, map_location=device))
        model.eval()
        model.requires_grad_(False)
        print(f"loaded ddlp model from {ddlp_ckpt}")
        print(f"particle normalizer: loaded ddlp model from {ddlp_ckpt}")
        seq_len = 50 if self.ds == 'traffic' else 100
        ds = get_video_dataset(self.ds, root=self.diffusion_config['ds_root'], mode='train', seq_len=seq_len)
        dl = DataLoader(ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
        pbar = tqdm(iterable=dl)
        z_all = []
        z_bg_all = []
        for i, batch in enumerate(pbar):
            x = batch[0][:, :self.diffusion_config['diffuse_frames']].to(device)
            x_prior = x
            batch_size, timesteps, ch, h, w = x.shape
            fg_dict = model.fg_sequential_opt(x, deterministic=True, x_prior=x, reshape=True)
            # encoder
            z = fg_dict['z']
            z_features = fg_dict['z_features']
            z_obj_on = fg_dict['obj_on']
            z_depth = fg_dict['z_depth']
            z_scale = fg_dict['z_scale']

            # decoder
            bg_mask = fg_dict['bg_mask']

            x_in = x.view(-1, *x.shape[2:])  # [bs * T, ...]
            bg_dict = model.bg_module(x_in, bg_mask, deterministic=True)
            z_bg = bg_dict['z_bg']
            z_kp_bg = bg_dict['z_kp']

            # collect and pad
            z_fg = torch.cat([z, z_scale, z_depth, z_obj_on.unsqueeze(-1), z_features], dim=-1)
            # [batch_size * timesteps, n_kp, features]
            z_fg = z_fg.view(-1, *z_fg.shape[2:])
            # [batch_size * timesteps * n_kp, features]
            z_all.append(z_fg.data.cpu())
            z_bg_all.append(z_bg.data.cpu())

        pbar.close()
        z_all = torch.cat(z_all, dim=0)
        z_bg_all = torch.cat(z_bg_all, dim=0)
        self.mu = z_all.mean(0)
        self.std = z_all.std(0)
        self.min_val = z_all.min(0)[0]
        self.max_val = z_all.max(0)[0]

        self.mu_bg = z_bg_all.mean(0)
        self.std_bg = z_bg_all.std(0)
        self.min_val_bg = z_bg_all.min(0)[0]
        self.max_val_bg = z_bg_all.max(0)[0]
        stats_path = os.path.join(self.root, 'latent_stats.pth')
        torch.save(self.state_dict(), stats_path)
        print(f'saved statistics @ {stats_path}')

    def normalize(self, z=None, z_bg=None):
        if self.mode == 'minmax':
            if z is not None:
                z = (z - self.min_val) / (self.max_val - self.min_val + self.eps)  # [0, 1]
                z = 2 * z - 1  # [-1, 1]
            if z_bg is not None:
                z_bg = (z_bg - self.min_val_bg) / (self.max_val_bg - self.min_val_bg + self.eps)  # [0, 1]
                z_bg = 2 * z_bg - 1  # [-1, 1]
        else:
            # std
            if z is not None:
                z = (z - self.mu) / (self.std + self.eps)
            if z_bg is not None:
                z_bg = (z_bg - self.mu_bg) / (self.std_bg + self.eps)

        return z, z_bg

    def unnormalize(self, z=None, z_bg=None):
        if self.mode == 'minmax':
            if z is not None:
                z = (z + 1) / 2  # [0, 1]
                z = z * (self.max_val - self.min_val + self.eps) + self.min_val
            if z_bg is not None:
                z_bg = (z_bg + 1) / 2  # [0, 1]
                z_bg = z_bg * (self.max_val_bg - self.min_val_bg + self.eps) + self.min_val_bg
        else:
            # std
            if z is not None:
                z = z * (self.std + self.eps) + self.mu
            if z_bg is not None:
                z_bg = z_bg * (self.std_bg + self.eps) + self.mu_bg

        return z, z_bg

    def forward(self, z=None, z_bg=None, normalize=True):
        if normalize:
            return self.normalize(z, z_bg)
        else:
            return self.unnormalize(z, z_bg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion DDLP Trainer")
    parser.add_argument("-c", "--config", type=str, default='diffuse_ddlp',
                        help="json file name of config file in './configs'")
    args = parser.parse_args()
    # parse input
    conf = args.config
    if conf.endswith('json'):
        conf_path = os.path.join('./configs', conf)
    else:
        conf_path = os.path.join('./configs', f'{conf}.json')
    diffusion_config = get_config(conf_path)
    ds = diffusion_config['ds']
    ds_root = diffusion_config['ds_root']  # dataset root
    batch_size = diffusion_config['batch_size']
    diffuse_frames = diffusion_config['diffuse_frames']  # number of particle frames to generate
    lr = diffusion_config['lr']
    train_num_steps = diffusion_config['train_num_steps']
    diffusion_num_steps = diffusion_config['diffusion_num_steps']
    loss_type = diffusion_config['loss_type']
    particle_norm = diffusion_config['particle_norm']
    device = diffusion_config['device']
    if 'cuda' in device:
        device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    """
    load pre-trained DDLP
    """
    ddlp_dir = diffusion_config['ddlp_dir']
    ddlp_ckpt = diffusion_config['ddlp_ckpt']
    ddlp_conf = os.path.join(ddlp_dir, 'hparams.json')
    ddlp_config = get_config(ddlp_conf)
    # load model
    image_size = ddlp_config['image_size']
    ch = ddlp_config['ch']
    enc_channels = ddlp_config['enc_channels']
    prior_channels = ddlp_config['prior_channels']
    use_correlation_heatmaps = ddlp_config['use_correlation_heatmaps']
    enable_enc_attn = ddlp_config['enable_enc_attn']
    filtering_heuristic = ddlp_config['filtering_heuristic']
    animation_fps = ddlp_config["animation_fps"]

    model = ObjectDynamicsDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                              image_size=image_size, n_kp=ddlp_config['n_kp'],
                              learned_feature_dim=ddlp_config['learned_feature_dim'],
                              pad_mode=ddlp_config['pad_mode'],
                              sigma=ddlp_config['sigma'],
                              dropout=ddlp_config['dropout'], patch_size=ddlp_config['patch_size'],
                              n_kp_enc=ddlp_config['n_kp_enc'],
                              n_kp_prior=ddlp_config['n_kp_prior'], kp_range=ddlp_config['kp_range'],
                              kp_activation=ddlp_config['kp_activation'],
                              anchor_s=ddlp_config['anchor_s'],
                              use_resblock=ddlp_config['use_resblock'],
                              timestep_horizon=ddlp_config['timestep_horizon'],
                              predict_delta=ddlp_config['predict_delta'],
                              scale_std=ddlp_config['scale_std'],
                              offset_std=ddlp_config['offset_std'], obj_on_alpha=ddlp_config['obj_on_alpha'],
                              obj_on_beta=ddlp_config['obj_on_beta'], pint_heads=ddlp_config['pint_heads'],
                              pint_layers=ddlp_config['pint_layers'], pint_dim=ddlp_config['pint_dim'],
                              use_correlation_heatmaps=use_correlation_heatmaps,
                              enable_enc_attn=enable_enc_attn, filtering_heuristic=filtering_heuristic).to(device)
    model.load_state_dict(torch.load(ddlp_ckpt, map_location=device))
    model.eval()
    model.requires_grad_(False)
    print(f"loaded ddlp model from {ddlp_ckpt}")

    features_dim = 2 + 2 + 1 + 1 + ddlp_config['learned_feature_dim']
    # features: xy, scale_xy, depth, obj_on, particle features
    # total particles: n_kp + 1 for bg
    ddpm_feat_dim = features_dim

    denoiser_model = PINTDenoiser(features_dim, hidden_dim=ddlp_config['pint_dim'],
                                  projection_dim=ddlp_config['pint_dim'],
                                  n_head=ddlp_config['pint_heads'], n_layer=ddlp_config['pint_layers'],
                                  block_size=diffuse_frames, dropout=0.1,
                                  predict_delta=False, positional_bias=True, max_particles=ddlp_config['n_kp_enc'] + 1,
                                  self_condition=False,
                                  learned_sinusoidal_cond=False, random_fourier_features=False,
                                  learned_sinusoidal_dim=16).to(device)

    diffusion = GaussianDiffusionPINT(
        denoiser_model,
        seq_length=diffuse_frames,
        timesteps=diffusion_num_steps,  # number of steps
        sampling_timesteps=diffusion_num_steps,
        loss_type=loss_type,  # L1 or L2
        objective='pred_x0',
    ).to(device)

    particle_normalizer = ParticleNormalization(diffusion_config, mode=particle_norm).to(device)
    result_dir = diffusion_config.get('result_dir')
    if result_dir is None:
        run_name = f'{ds}_diffuse_ddlp'
        result_dir = prepare_logdir(run_name, src_dir='./')
        diffusion_config['result_dir'] = result_dir

    # make copy of configs
    path_to_conf = os.path.join(result_dir, 'ddlp_hparams.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(ddlp_config, outfile, indent=2)
    path_to_conf = os.path.join(result_dir, 'diffusion_hparams.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(diffusion_config, outfile, indent=2)
    latent_stats_path = os.path.join(ddlp_dir, 'latent_stats.pth')  # make a copy of latent stats just in case
    latent_stats_path_target = os.path.join(result_dir, 'latent_stats.pth')
    shutil.copy(latent_stats_path, latent_stats_path_target)

    # expects input: [batch_size, feature_dim, seq_len]

    trainer = TrainerDiffuseDDLP(
        diffusion,
        ddlp_model=model,
        diffusion_config=diffusion_config,
        particle_norm=particle_normalizer,
        train_batch_size=batch_size,
        train_lr=lr,
        train_num_steps=train_num_steps,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        seq_len=diffuse_frames,
        save_and_sample_every=1000,
        results_folder=result_dir, animation_fps=animation_fps
    )

    trainer.train()
