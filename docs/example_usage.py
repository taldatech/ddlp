"""
Example usage of DLPv2 and DDLP
"""
# imports
import os
import sys

sys.path.append(os.getcwd())
# torch
import torch
# modules
from models import ObjectDLP, ObjectDynamicsDLP

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # example hyper-parameters
    batch_size = 32
    beta_kl = 0.1
    beta_rec = 1.0
    kl_balance = 0.001  # balance between spatial attributes (x, y, scale, depth) and visual features
    n_kp_enc = 12
    n_kp_prior = 15
    patch_size = 8  # patch size for the prior to generate prior proposals
    learned_feature_dim = 6  # visual features
    anchor_s = 0.25  # effective patch size for the posterior: anchor_s * image_size

    image_size = 64
    ch = 3
    enc_channels = [32, 64, 128]
    prior_channels = (32, 32, 64)

    use_correlation_heatmaps = False  # for tracking, set True to use correlation heatmaps between patches

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    print("--- DLPv2 ---")

    # create model
    model = ObjectDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                      image_size=image_size, learned_feature_dim=learned_feature_dim,
                      patch_size=patch_size, n_kp_enc=n_kp_enc, n_kp_prior=n_kp_prior,
                      anchor_s=anchor_s, use_correlation_heatmaps=use_correlation_heatmaps).to(device)
    print(f'model.info():')
    print(model.info())
    print("----------------------------------")
    # dummy data
    x = torch.rand(batch_size, ch, image_size, image_size, device=device)
    # complete forward
    model_output = model(x)
    # let's see what's inside
    print(f'model(x) output:')
    for k in model_output.keys():
        print(f'{k}: {model_output[k].shape}')
    print("----------------------------------")
    """
    output:
    kp_p: torch.Size([32, 15, 2])  # prior proposals
    rec: torch.Size([32, 3, 64, 64])  # full reconstructions
    mu: torch.Size([32, 12, 2])  # position mu
    logvar: torch.Size([32, 12, 2])  # position logvar
    z: torch.Size([32, 12, 2])  # position z
    z_base: torch.Size([32, 12, 2])  # position anchors (mu = z_base + mu_offset)
    z_kp_bg: torch.Size([32, 1, 2])  # constants (0.0, 0.0) for the bg kp
    mu_offset: torch.Size([32, 12, 2])  # position offset mu
    logvar_offset: torch.Size([32, 12, 2])  # position offset logvar
    mu_features: torch.Size([32, 12, 6])  # visual features mu
    logvar_features: torch.Size([32, 12, 6])  # visual features logvar
    z_features: torch.Size([32, 12, 6])  # visual features z
    bg: torch.Size([32, 3, 64, 64])  # bg reconstructions
    mu_bg: torch.Size([32, 6])  # bg visual features mu
    logvar_bg: torch.Size([32, 6])  # bg visual features logvar
    z_bg: torch.Size([32, 6])  # bg visual features z
    cropped_objects_original: torch.Size([32, 12, 3, 16, 16])  # extracted patches from the original image
    obj_on_a: torch.Size([32, 12])  # transparency beta distribution "a" parameter
    obj_on_b: torch.Size([32, 12])  # transparency beta distribution "b" parameter
    obj_on: torch.Size([32, 12])  # transparency sample per particle
    dec_objects_original: torch.Size([32, 12, 4, 16, 16])  # decoded glimpses (rgb + alpha channel)
    dec_objects: torch.Size([32, 3, 64, 64])  # decoded foreground (no bg)
    mu_depth: torch.Size([32, 12, 1])  # depth mu
    logvar_depth: torch.Size([32, 12, 1])  # depth logvar
    z_depth: torch.Size([32, 12, 1])  # depth z
    mu_scale: torch.Size([32, 12, 2])  # scale mu
    logvar_scale: torch.Size([32, 12, 2])  # scale logvar
    z_scale: torch.Size([32, 12, 2])  # scale z
    alpha_masks: torch.Size([32, 12, 1, 64, 64])  # objects masks
    """

    # loss calculation
    all_losses = model.calc_elbo(x, model_output, beta_kl=beta_kl,
                                 beta_rec=beta_rec, kl_balance=kl_balance,
                                 recon_loss_type="mse")
    # let's see what's inside
    print(f'model.calc_elbo(): model losses:')
    for k in all_losses.keys():
        print(f'{k}: {all_losses[k]}')
    print("----------------------------------")
    """
    output:
    loss: the complete loss (for loss.backward())
    psnr: mean PSNR
    kl: complete kl-divergence (of all components)
    loss_rec: reconstruction loss
    obj_on_l1: if all particles are "on" then obj_on_l1=n_particles, effective # of visible particles
    loss_kl_kp: kl of the position
    loss_kl_feat: kl of the visual features
    loss_kl_obj_on: kl of the transparency
    loss_kl_scale: kl of the scale
    loss_kl_depth: kl of the depth
    """

    # only encoding:
    model_output = model.encode_all(x, deterministic=True)  # deterministic=True -> z = mu
    # let's see what's inside
    print(f'model.encode_all(): model encoder output:')
    for k in model_output.keys():
        out_print = model_output[k].shape if model_output[k] is not None else None
        print(f'{k}: {out_print}')
    print("----------------------------------")
    """
    output:
    mu: torch.Size([32, 12, 2])
    logvar: torch.Size([32, 12, 2])
    z: torch.Size([32, 12, 2])
    z_base: torch.Size([32, 12, 2])
    kp_heatmap: None  # this is not used in this model, it was used in the non-object DLP model 
    mu_features: torch.Size([32, 12, 6])
    logvar_features: torch.Size([32, 12, 6])
    z_features: torch.Size([32, 12, 6])
    obj_on_a: torch.Size([32, 12])
    obj_on_b: torch.Size([32, 12])
    obj_on: torch.Size([32, 12])
    mu_depth: torch.Size([32, 12, 1])
    logvar_depth: torch.Size([32, 12, 1])
    z_depth: torch.Size([32, 12, 1])
    cropped_objects: torch.Size([32, 12, 3, 16, 16])
    bg_mask: torch.Size([32, 1, 64, 64])
    mu_scale: torch.Size([32, 12, 2])
    logvar_scale: torch.Size([32, 12, 2])
    z_scale: torch.Size([32, 12, 2])
    mu_offset: torch.Size([32, 12, 2])
    logvar_offset: torch.Size([32, 12, 2])
    z_offset: torch.Size([32, 12, 2])
    mu_bg: torch.Size([32, 6])
    logvar_bg: torch.Size([32, 6])
    z_bg: torch.Size([32, 6])
    z_kp_bg: torch.Size([32, 1, 2])
    """

    # only decoding:
    z = model_output['z']
    z_scale = model_output['z_scale']
    z_depth = model_output['z_depth']
    z_obj_on = model_output['obj_on']
    z_features = model_output['z_features']
    z_bg = model_output['z_bg']

    decode_output = model.decode_all(z=z, z_features=z_features, z_bg=z_bg, obj_on=z_obj_on, z_scale=z_scale,
                                     z_depth=z_depth)
    # let's see what's inside
    print(f'model.decode_all(): model decode output:')
    for k in decode_output.keys():
        out_print = decode_output[k].shape if decode_output[k] is not None else None
        print(f'{k}: {out_print}')
    print("----------------------------------")
    """
    output:
    rec: torch.Size([32, 3, 64, 64])
    dec_objects: torch.Size([32, 12, 4, 16, 16])  # decoded glimpses (rgb + alpha channel)
    dec_objects_trans: torch.Size([32, 3, 64, 64])  # decoded foreground (no bg)
    bg: torch.Size([32, 3, 64, 64])
    alpha_masks: torch.Size([32, 12, 1, 64, 64])
    """

    print("--- DLPv2 with Tracking ---")
    # tracking
    use_correlation_heatmaps = True
    use_tracking = True
    model = ObjectDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                      image_size=image_size, learned_feature_dim=learned_feature_dim,
                      patch_size=patch_size, n_kp_enc=n_kp_enc, n_kp_prior=n_kp_prior,
                      anchor_s=anchor_s,
                      use_correlation_heatmaps=use_correlation_heatmaps, use_tracking=use_tracking).to(device)
    num_frames = 3
    x = torch.rand(1, num_frames, ch, image_size, image_size, device=device)
    model_output = model(x)
    # let's see what's inside
    print(f'model(x) tracking output:')
    for k in model_output.keys():
        print(f'{k}: {model_output[k].shape}')
    print("----------------------------------")
    """
    output:
    similar to before, but the first dimension is batch_size * num_frames for all
    """

    print("--- DDLP ---")
    # example additional hyper-parameters
    use_correlation_heatmaps = True
    pint_layers = 6  # transformer-based dynamics module number of layers
    pint_heads = 8  # transformer-based dynamics module attention heads
    pint_dim = 256  # transformer-based dynamics module inner dimension (+projection dim)
    beta_dyn = 0.1  # beta-kl for the dynamics loss
    num_static_frames = 4  # "burn-in frames", number of initial frames with kl w.r.t. constant prior (as in DLPv2)

    model = ObjectDynamicsDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                              image_size=image_size, learned_feature_dim=learned_feature_dim,
                              patch_size=patch_size, n_kp_enc=n_kp_enc, n_kp_prior=n_kp_prior,
                              anchor_s=anchor_s, use_correlation_heatmaps=use_correlation_heatmaps,
                              pint_layers=pint_layers, pint_heads=pint_heads, pint_dim=pint_dim).to(device)
    print(f'model.info():')
    print(model.info())
    print("----------------------------------")
    timestep_horizon = 10
    x = torch.rand(batch_size, timestep_horizon + 1, ch, image_size, image_size, device=device)
    model_output = model(x)
    # let's see what's inside
    print(f'model(x) output:')
    for k in model_output.keys():
        print(f'{k}: {model_output[k].shape}')
    print("----------------------------------")
    """
    output: similar to before, but the first dimension is batch_size * num_frames for all
    kp_p: torch.Size([352, 15, 2])
    rec: torch.Size([352, 3, 64, 64])
    mu: torch.Size([352, 12, 2])
    logvar: torch.Size([352, 12, 2])
    z_base: torch.Size([352, 12, 2])
    z: torch.Size([352, 12, 2])
    z_kp_bg: torch.Size([352, 1, 2])
    mu_offset: torch.Size([352, 12, 2])
    logvar_offset: torch.Size([352, 12, 2])
    mu_features: torch.Size([352, 12, 6])
    logvar_features: torch.Size([352, 12, 6])
    z_features: torch.Size([352, 12, 6])
    bg: torch.Size([352, 3, 64, 64])
    mu_bg: torch.Size([352, 6])
    logvar_bg: torch.Size([352, 6])
    z_bg: torch.Size([352, 6])
    cropped_objects_original: torch.Size([352, 12, 3, 16, 16])
    obj_on_a: torch.Size([352, 12])
    obj_on_b: torch.Size([352, 12])
    obj_on: torch.Size([352, 12])
    dec_objects_original: torch.Size([352, 12, 4, 16, 16])
    dec_objects: torch.Size([352, 3, 64, 64])
    mu_depth: torch.Size([352, 12, 1])
    logvar_depth: torch.Size([352, 12, 1])
    z_depth: torch.Size([352, 12, 1])
    mu_scale: torch.Size([352, 12, 2])
    logvar_scale: torch.Size([352, 12, 2])
    z_scale: torch.Size([352, 12, 2])
    alpha_masks: torch.Size([352, 12, 1, 64, 64])
    mu_dyn: torch.Size([32, 10, 12, 2])  # dynamics-prior position for t=1->T-1 given t=0->T-2
    logvar_dyn: torch.Size([32, 10, 12, 2])  # dynamics-prior position for t=1->T-1 given t=0->T-2
    mu_features_dyn: torch.Size([32, 10, 12, 6])  # dynamics-prior visual appearance for t=1->T-1 given t=0->T-2
    logvar_features_dyn: torch.Size([32, 9, 12, 6])  # dynamics-prior visual appearance for t=1->T-1 given t=0->T-2
    obj_on_a_dyn: torch.Size([32, 10, 12])  # dynamics-prior transparency for t=1->T-1 given t=0->T-2
    obj_on_b_dyn: torch.Size([32, 10, 12])  # dynamics-prior transparency for t=1->T-1 given t=0->T-2
    mu_depth_dyn: torch.Size([32, 10, 12, 1])  # dynamics-prior depth for t=1->T-1 given t=0->T-2
    logvar_depth_dyn: torch.Size([32, 10, 12, 1])  # dynamics-prior depth for t=1->T-1 given t=0->T-2
    mu_scale_dyn: torch.Size([32, 10, 12, 2])  # dynamics-prior scale for t=1->T-1 given t=0->T-2
    logvar_scale_dyn: torch.Size([32, 10, 12, 2])  # dynamics-prior scale for t=1->T-1 given t=0->T-2
    mu_bg_dyn: torch.Size([32, 10, 6])  # dynamics-prior background appearance for t=1->T-1 given t=0->T-2
    logvar_bg_dyn: torch.Size([32, 10, 6])  # dynamics-prior background appearance for t=1->T-1 given t=0->T-2
    """

    # loss calculation
    all_losses = model.calc_elbo(x, model_output, beta_kl=beta_kl,
                                 beta_rec=beta_rec, kl_balance=kl_balance, beta_dyn=beta_dyn,
                                 num_static=num_static_frames,
                                 recon_loss_type="mse")
    # let's see what's inside
    print(f'model.calc_elbo(): model losses:')
    for k in all_losses.keys():
        print(f'{k}: {all_losses[k]}')
    print("----------------------------------")
    """
    output:
    model.calc_elbo(): model losses:
    loss: 465.488525390625
    psnr: 10.788384437561035
    kl: 2579.010986328125
    kl_dyn: 406.3466491699219  # <---- dynamics kl
    loss_rec: 4821.837890625
    obj_on_l1: 5.660502910614014
    loss_kl_kp: 997.2069702148438
    loss_kl_feat: 0.212782084941864
    loss_kl_obj_on: 56.031768798828125
    loss_kl_scale: 1525.77197265625
    loss_kl_depth: 0.08919306844472885
    """

    # sampling
    num_steps = 15
    cond_steps = 5
    x = torch.rand(1, num_steps + cond_steps, ch, image_size, image_size, device=device)
    sample_out, sample_z_out = model.sample(x, cond_steps=cond_steps, num_steps=num_steps, deterministic=True,
                                            return_z=True)
    # let's see what's inside
    print(f'model.sample(): model dynamics unrolling:')
    print(f'sample_out: {sample_out.shape}')
    print(f'sample_z_out:')
    for k in sample_z_out.keys():
        print(f'{k}: {sample_z_out[k].shape}')
    print("----------------------------------")
    """
    output:
    sample_out: torch.Size([1, 20, 3, 64, 64])  #  generated frames
    sample_z_out:  # latent unrolls
    z_pos: torch.Size([1, 20, 12, 2])
    z_scale: torch.Size([1, 20, 12, 2])
    z_obj_on: torch.Size([1, 20, 12])
    z_depth: torch.Size([1, 20, 12, 1])
    z_features: torch.Size([1, 20, 12, 6])
    z_bg_features: torch.Size([1, 20, 6])
    z_ids: torch.Size([1, 20, 12])  #  this is only used for the balls-interaction dataset, each particle gets an id
    """
