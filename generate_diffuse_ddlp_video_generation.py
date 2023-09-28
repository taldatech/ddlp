"""
Script to generate unconditional videos/images from a pre-trained DiffuseDDLP
"""

# imports
import os
import argparse

# torch
import torch

# utils
from utils.util_func import get_config

# models
from modules.diffusion_modules import TrainerDiffuseDDLP, GaussianDiffusionPINT, PINTDenoiser
from train_diffuse_ddlp import ParticleNormalization
from models import ObjectDynamicsDLP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion DDLP Video Generation")
    parser.add_argument("-c", "--config", type=str,
                        help="json file name of config file in the pre-trained model dir")
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=10)
    parser.add_argument("-n", "--num_samples", type=int, help="num samples to generate", default=30)
    parser.add_argument("--cpu", action='store_true', help="use cpu for inference")
    parser.add_argument("--image", action='store_true', help="generate image plot instead of video")

    args = parser.parse_args()
    # parse input
    conf_path = args.config
    diffusion_config = get_config(conf_path)
    batch_size = args.batch_size
    num_samples = args.num_samples
    use_cpu = args.cpu
    gen_img = args.image
    result_dir = diffusion_config['result_dir']
    ds = diffusion_config['ds']
    ds_root = diffusion_config['ds_root']  # dataset root
    diffuse_frames = diffusion_config['diffuse_frames']  # number of particle frames to generate
    lr = diffusion_config['lr']
    train_num_steps = diffusion_config['train_num_steps']
    diffusion_num_steps = diffusion_config['diffusion_num_steps']
    loss_type = diffusion_config['loss_type']
    particle_norm = diffusion_config['particle_norm']
    device = "cpu" if use_cpu else diffusion_config['device']

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

    trainer.load()
    if gen_img:
        trainer.sample_image(num_samples)
    else:
        trainer.sample(num_samples)
