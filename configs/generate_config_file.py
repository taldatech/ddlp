import json
import os


def save_config(src_dir, fname, hparams):
    path_to_conf = os.path.join(src_dir, fname)
    with open(path_to_conf, "w") as outfile:
        json.dump(hparams, outfile, indent=2)


def gen_conf_file(ds, fname='default.json'):
    device = 'cuda'
    lr = 2e-4
    batch_size = 32
    num_epochs = 150
    load_model = False
    pretrained_path = None
    eval_epoch_freq = 1
    n_kp = 1  # num kp per patch
    iou_thresh = 0.2
    kp_range = (-1, 1)
    weight_decay = 0.0
    run_prefix = ""
    pad_mode = 'replicate'
    sigma = 1.0  # default sigma for the gaussian maps
    dropout = 0.0
    kp_activation = "tanh"
    ch = 3  # image channels
    topk = 5  # top-k particles to plot
    use_resblock = False
    adam_betas = (0.9, 0.999)
    adam_eps = 1e-4
    # filtering heuristic to filter prior keypoints
    filtering_heuristic = 'variance'  # ['distance', 'variance', 'random', 'none']

    timestep_horizon = 10
    predict_delta = True
    use_tracking = True
    use_correlation_heatmaps = True  # use correlation heatmap between patches for tracking
    enable_enc_attn = False  # rnable attention between patches in the particle encoder

    beta_kl = 0.1
    beta_dyn = 0.1
    beta_rec = 1.0
    beta_dyn_rec = 1.0
    kl_balance = 0.001

    num_static_frames = 4

    pint_layers = 6  # transformer layers in the dynamics module
    pint_dim = 256
    pint_heads = 8

    # priors
    scale_std = 0.3
    offset_std = 0.2
    obj_on_alpha = 0.1
    obj_on_beta = 0.1

    animation_horizon = 100
    eval_im_metrics = True
    scheduler_gamma = 0.95
    train_enc_prior = True  # train the SSM prior or leave random
    start_dyn_epoch = 0  # epoch from which to start training the dynamics module
    cond_steps = 10  # conditional steps for the dynamics module during inference
    animation_fps = 3 / 50

    if ds == 'traffic':
        beta_kl = 40.0
        beta_dyn = 40.0
        beta_rec = 1.0
        beta_dyn_rec = 1.0
        # n_kp_enc = 16  # total kp to output from the encoder / filter from prior
        n_kp_enc = 25
        # n_kp_prior = 20
        n_kp_prior = 30
        patch_size = 16
        learned_feature_dim = 8  # additional features than x,y for each kp
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "vgg"
        # warmup_epoch = 1
        warmup_epoch = 3
        anchor_s = 0.25
        kl_balance = 0.001
        exclusive_patches = False
        # batch_size = 2
        batch_size = 8  # a100
        eval_epoch_freq = 1
        # timestep_horizon = 20
        timestep_horizon = 10
        lr = 2e-4
        # lr = 5e-4
        image_size = 128
        enc_channels = [32, 64, 128, 256]
        prior_channels = (16, 32, 64)
        # root = '/mnt/data/tal/traffic_dataset/img128np_fs3.npy'
        root = '/home/tal/data/traffic/img128np.npy'
    elif ds == 'clevrer':
        beta_kl = 100.0
        beta_dyn = 100.0
        beta_rec = 1.0
        beta_dyn_rec = 1.0
        # n_kp_enc = 10  # total kp to output from the encoder / filter from prior
        n_kp_enc = 12  # total kp to output from the encoder / filter from prior
        # n_kp_prior = 20
        n_kp_prior = 16  # orig
        # n_kp_prior = 10
        patch_size = 16
        learned_feature_dim = 8  # additional features than x,y for each kp
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "vgg"
        # recon_loss_type = "mse"
        warmup_epoch = 1
        # warmup_epoch = 0
        anchor_s = 0.25
        kl_balance = 0.001
        # batch_size = 4
        batch_size = 8
        eval_epoch_freq = 1
        timestep_horizon = 20
        lr = 2e-4
        image_size = 128
        # image_size = 64
        enc_channels = [32, 64, 128, 256]  # 128x128
        # enc_channels = [32, 64, 128]  # 64x64
        prior_channels = (16, 32, 64)
        # root = '/mnt/data/tal/clevrer_ep/'
        root = '/datadrive/clevrer/'
    elif ds == 'balls':
        beta_kl = 0.1  # original 0.05
        beta_dyn = 0.1  # original 0.1
        beta_rec = 1.0
        beta_dyn_rec = 1.0
        # beta_rec = 1 / 11
        # n_kp_enc = 3  # total kp to output from the encoder / filter from prior
        n_kp_enc = 6
        n_kp_prior = 12  # original
        # n_kp_prior = 15
        patch_size = 8  # original
        # patch_size = 16
        learned_feature_dim = 3  # additional features than x,y for each kp
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "mse"
        warmup_epoch = 1
        # warmup_epoch = -2
        anchor_s = 0.25
        kl_balance = 0.001
        # override manually
        # lr = 2e-4
        batch_size = 32
        # batch_size = 20
        eval_epoch_freq = 1
        predict_delta = True
        image_size = 64
        ch = 3
        enc_channels = (32, 64, 128)
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/gswm_balls/BALLS_INTERACTION'
    elif ds == 'bair':
        beta_kl = 40.0
        beta_dyn = 800.0
        beta_rec = 1.0
        beta_dyn_rec = 1.0
        # n_kp_enc = 3  # total kp to output from the encoder / filter from prior
        n_kp_enc = 15
        n_kp_prior = 20  # original
        # n_kp_prior = 15
        patch_size = 8  # original
        # patch_size = 16
        learned_feature_dim = 10  # additional features than x,y for each kp
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "vgg"
        warmup_epoch = 1
        # warmup_epoch = 0
        anchor_s = 0.25
        kl_balance = 0.001
        # override manually
        lr = 2e-4
        batch_size = 8
        eval_epoch_freq = 1
        timestep_horizon = 15
        image_size = 64
        enc_channels = (32, 64, 128)
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/bair/processed/'
        # root = '/media/newhd/data/bair/processed/'
        # dataset = BAIRDataset(root=root, train=True, horizon=timestep_horizon + 1)
        animation_horizon = 16
        cond_steps = 1
    elif ds == 'obj3d':
        # mse:
        # beta_kl = 0.01  # original: 0.05, worked good: 0.01
        # beta_dyn = 0.01
        # beta_rec = 1.0
        # vgg:
        beta_kl = 30.0
        beta_dyn = 30.0
        beta_rec = 1.0
        beta_dyn_rec = 1.0
        # n_kp_enc = 3  # total kp to output from the encoder / filter from prior
        n_kp_enc = 8
        n_kp_prior = 16  # original
        # n_kp_prior = 15
        patch_size = 8  # original
        # patch_size = 16
        learned_feature_dim = 10  # additional features than x,y for each kp
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        # recon_loss_type = "mse"
        recon_loss_type = "vgg"
        warmup_epoch = 1
        # warmup_epoch = 0
        anchor_s = 0.25
        kl_balance = 0.001
        # override manually
        lr = 2e-4
        # batch_size = 20  # mse
        batch_size = 10  # vgg
        eval_epoch_freq = 1
        image_size = 64
        enc_channels = (32, 64, 128)
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/obj3d/'
    elif ds == 'obj3d128':
        # mse:
        # beta_kl = 0.1  # original: 0.05, worked good: 0.01
        # beta_dyn = 0.1
        # beta_rec = 1.0
        # vgg:
        beta_kl = 100.0
        beta_dyn = 100.0
        beta_rec = 1.0
        beta_dyn_rec = 1.0

        # recon_loss_type = "mse"
        recon_loss_type = "vgg"

        # beta_rec = 1 / 11
        # n_kp_enc = 3  # total kp to output from the encoder / filter from prior
        n_kp_enc = 12
        n_kp_prior = 16  # original
        # n_kp_prior = 64
        # patch_size = 8  # original
        patch_size = 16
        # learned_feature_dim = 10  # additional features than x,y for each kp
        learned_feature_dim = 8
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance

        warmup_epoch = 1
        # warmup_epoch = 0
        anchor_s = 0.25
        kl_balance = 0.001
        # override manually
        lr = 2e-4
        # batch_size = 6  # mse
        batch_size = 4  # vgg
        eval_epoch_freq = 1
        timestep_horizon = 10
        # sigma = 1.0  # deterministic chamfer
        image_size = 128
        enc_channels = [32, 64, 128, 256]  # 128x128
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/obj3d/'
    elif ds == 'sketchy':
        # mse:
        # beta_kl = 0.01  # original: 0.05, worked good: 0.01
        # beta_dyn = 0.01
        # beta_rec = 1.0
        # vgg:
        beta_kl = 40.0
        beta_dyn = 40.0
        beta_rec = 1.0
        beta_dyn_rec = 1.0
        # n_kp_enc = 3  # total kp to output from the encoder / filter from prior
        n_kp_enc = 8
        n_kp_prior = 16  # original
        # n_kp_prior = 15
        # patch_size = 8  # original
        patch_size = 16
        learned_feature_dim = 10  # additional features than x,y for each kp
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        # recon_loss_type = "mse"
        recon_loss_type = "vgg"
        warmup_epoch = 1
        # warmup_epoch = 0
        anchor_s = 0.25
        kl_balance = 0.001
        # override manually
        lr = 2e-4
        # batch_size = 20  # mse
        batch_size = 4  # vgg
        eval_epoch_freq = 1
        sigma = 1.0  # deterministic chamfer
        image_size = 128
        enc_channels = [32, 64, 128, 256]  # 128x128
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/sketchy/'
    elif ds == 'phyre':
        # mse:
        beta_kl = 0.15  # orig: 0.1
        beta_dyn = 0.15  # prig: 0.1
        beta_rec = 1.0
        beta_dyn_rec = 1.0
        # vgg:
        # beta_kl = 20.0
        # beta_dyn = 40.0
        # beta_rec = 1.0
        # beta_dyn_rec = 0.1  # 0.1
        n_kp_enc = 25  # anchor_s:0.125
        n_kp_prior = 30  # anchor_s:0.125
        # n_kp_enc = 15  # anchor_s:0.25
        # n_kp_prior = 20  # anchor_s:0.25
        # n_kp_prior = 16  # original
        # patch_size = 8  # original
        patch_size = 16  # TODO: 8?
        learned_feature_dim = 4  # additional features than x,y for each kp
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "mse"
        # recon_loss_type = "vgg"
        warmup_epoch = 1
        # warmup_epoch = -2
        anchor_s = 0.125
        # anchor_s = 0.25
        kl_balance = 0.001
        # override manually
        # lr = 2e-4
        # batch_size = 6  # mse, t10
        batch_size = 3  # mse, t20
        # batch_size = 4  # vgg 64
        # batch_size = 2  # vgg 128
        eval_epoch_freq = 1
        sigma = 1.0  # deterministic chamfer
        predict_delta = True
        # timestep_horizon = 10
        timestep_horizon = 15
        pint_dim = 512
        image_size = 128
        # image_size = 64
        enc_channels = [32, 64, 128, 256]  # 128x128
        # enc_channels = [32, 64, 128]  # 64x64
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/phyre/'
        # root = '/media/newhd/data/phyre/'
        animation_fps = 2.5 / 50
    elif ds == 'mario':
        # mse:
        # beta_kl = 0.01  # original: 0.05, worked good: 0.01
        # beta_dyn = 0.01
        # beta_rec = 1.0
        # vgg:
        beta_kl = 80.0
        beta_dyn = 80.0
        beta_rec = 1.0
        beta_dyn_rec = 1.0
        # n_kp_enc = 3  # total kp to output from the encoder / filter from prior
        n_kp_enc = 15
        n_kp_prior = 20  # original
        # n_kp_prior = 15
        # patch_size = 8  # original
        patch_size = 16
        learned_feature_dim = 10  # additional features than x,y for each kp
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        # recon_loss_type = "mse"
        recon_loss_type = "vgg"
        warmup_epoch = 1
        # warmup_epoch = 0
        anchor_s = 0.25
        kl_balance = 0.001
        # override manually
        lr = 2e-4
        # batch_size = 20  # mse
        batch_size = 4  # vgg
        eval_epoch_freq = 1
        image_size = 128
        enc_channels = [32, 64, 128, 256]
        prior_channels = (16, 32, 64)
        root = '/media/newhd/data/mario/'
    elif ds == 'shapes':
        beta_kl = 0.1  # original
        beta_rec = 1.0
        n_kp_enc = 10  # total kp to output from the encoder / filter from prior
        n_kp_prior = 64
        patch_size = 8
        learned_feature_dim = 6  # additional features than x,y for each kp
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "mse"
        warmup_epoch = 1
        eval_epoch_freq = 1
        anchor_s = 0.25
        kl_balance = 0.001
        lr = 1e-3
        batch_size = 64
        image_size = 64
        enc_channels = (32, 64, 128)
        prior_channels = (16, 32, 64)
        root = None
        filtering_heuristic = 'none'
        use_correlation_heatmaps = False
        use_tracking = False
    else:
        raise NotImplementedError("unrecognized dataset, please implement it and add it to the train script")

    hparams = {'ds': ds, 'root': root, 'device': device, 'batch_size': batch_size, 'lr': lr,
               'kp_activation': kp_activation,
               'pad_mode': pad_mode, 'load_model': load_model, 'pretrained_path': pretrained_path,
               'num_epochs': num_epochs, 'n_kp': n_kp, 'recon_loss_type': recon_loss_type,
               'sigma': sigma, 'beta_kl': beta_kl, 'beta_rec': beta_rec,
               'patch_size': patch_size, 'topk': topk, 'n_kp_enc': n_kp_enc,
               'eval_epoch_freq': eval_epoch_freq, 'learned_feature_dim': learned_feature_dim,
               'n_kp_prior': n_kp_prior, 'weight_decay': weight_decay, 'kp_range': kp_range,
               'warmup_epoch': warmup_epoch, 'dropout': dropout,
               'iou_thresh': iou_thresh, 'anchor_s': anchor_s, 'kl_balance': kl_balance,
               'image_size': image_size, 'ch': ch, 'enc_channels': enc_channels,
               'prior_channels': prior_channels,
               'timestep_horizon': timestep_horizon, 'predict_delta': predict_delta, 'beta_dyn': beta_dyn,
               'scale_std': scale_std, 'offset_std': offset_std, 'obj_on_alpha': obj_on_alpha,
               'obj_on_beta': obj_on_beta, 'beta_dyn_rec': beta_dyn_rec, 'num_static_frames': num_static_frames,
               'pint_layers': pint_layers, 'pint_heads': pint_heads, 'pint_dim': pint_dim, 'run_prefix': run_prefix,
               'animation_horizon': animation_horizon, 'eval_im_metrics': eval_im_metrics, 'use_resblock': use_resblock,
               'scheduler_gamma': scheduler_gamma, 'adam_betas': adam_betas, 'adam_eps': adam_eps,
               'train_enc_prior': train_enc_prior, 'start_dyn_epoch': start_dyn_epoch, 'cond_steps': cond_steps,
               'animation_fps': animation_fps, 'use_correlation_heatmaps': use_correlation_heatmaps,
               'enable_enc_attn': enable_enc_attn, 'filtering_heuristic': filtering_heuristic,
               'use_tracking': use_tracking}

    save_config('./', fname, hparams)


if __name__ == '__main__':
    # dss = ['traffic', 'balls', 'clevrer', 'phyre', 'obj3d', 'obj3d128', 'mario', 'sketchy', 'bair']
    dss = ['shapes']
    for ds in dss:
        gen_conf_file(ds=ds, fname=f'{ds}.json')
        conf_path = os.path.join('', f'{ds}.json')
        with open(conf_path, 'r') as f:
            config = json.load(f)
        print(config)
