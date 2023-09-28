"""
Evaluate image metrics such as LPIPS, PSNR and SSIM using PIQA,
"""

# set workdir
import os
import sys

sys.path.append(os.getcwd())
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import json
from tqdm import tqdm
from models import ObjectDynamicsDLP
# datasets
from datasets.get_dataset import get_video_dataset, get_image_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from piqa import PSNR, LPIPS, SSIM
except ImportError:
    print("piqa library required to compute image metrics")
    raise SystemExit


class ImageMetrics(nn.Module):
    """
    A class to calculate visual metrics between generated and ground-truth images
    """

    def __init__(self, metrics=('ssim', 'psnr', 'lpips')):
        super().__init__()
        self.metrics = metrics
        self.ssim = SSIM(reduction='none') if 'ssim' in self.metrics else None
        self.psnr = PSNR(reduction='none') if 'psnr' in self.metrics else None
        self.lpips = LPIPS(network='vgg', reduction='none') if 'lpips' in self.metrics else None

    @torch.no_grad()
    def forward(self, x, y):
        # x, y: [batch_size, 3, im_size, im_size] in [0,1]
        results = {}
        if self.ssim is not None:
            results['ssim'] = self.ssim(x, y)
        if self.psnr is not None:
            results['psnr'] = self.psnr(x, y)
        if self.lpips is not None:
            results['lpips'] = self.lpips(x, y)
        return results


def eval_ddlp_im_metric(model, device, config, timestep_horizon=50, val_mode='val', eval_dir='./',
                        cond_steps=10,
                        metrics=('ssim', 'psnr', 'lpips'), batch_size=32, verbose=False, accelerator=None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon, mode=val_mode, image_size=image_size)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False)
    model_timestep_horizon = model.timestep_horizon
    cond_steps = model_timestep_horizon if cond_steps is None else cond_steps

    # image metric instance
    evaluator = ImageMetrics(metrics=metrics).to(device)

    results = {}
    ssims = []
    psnrs = []
    lpipss = []
    for i, batch in enumerate(tqdm(dataloader)):
        x = batch[0][:, :timestep_horizon].to(device)
        with torch.no_grad():
            generated = model.sample(x, cond_steps=cond_steps, num_steps=timestep_horizon - cond_steps)
            generated = generated.clamp(0, 1)
            assert x.shape[1] == generated.shape[1], "prediction and gt frames shape don't match"
            results = evaluator(x[:, cond_steps:].reshape(-1, *x.shape[2:]),
                                generated[:, cond_steps:].reshape(-1, *generated.shape[2:]))
        # [batch_size * T]
        if 'ssim' in metrics:
            ssims.append(results['ssim'])
        if 'psnr' in metrics:
            psnrs.append(results['psnr'])
        if 'lpips' in metrics:
            lpipss.append(results['lpips'])

    if 'ssim' in metrics:
        ssims = torch.cat(ssims, dim=0)
        mean_ssim = ssims.mean().data.cpu().item()
        std_ssim = ssims.std().data.cpu().item()
        results['ssim'] = mean_ssim
        results['ssim_std'] = std_ssim
    if 'psnr' in metrics:
        psnrs = torch.cat(psnrs, dim=0)
        mean_psnr = psnrs.mean().data.cpu().item()
        std_psnr = psnrs.std().data.cpu().item()
        results['psnr'] = mean_psnr
        results['psnr_std'] = std_psnr
    if 'lpips' in metrics:
        lpipss = torch.cat(lpipss, dim=0)
        mean_lpips = lpipss.mean().data.cpu().item()
        std_lpips = lpipss.std().data.cpu().item()
        results['lpips'] = mean_lpips
        results['lpips_std'] = std_lpips

    # save results
    path_to_conf = os.path.join(eval_dir, 'last_val_image_metrics.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(results, outfile, indent=2)

    del evaluator  # clear memory

    return results


def eval_dlp_im_metric(model, device, config, val_mode='val', eval_dir='./',
                       metrics=('ssim', 'psnr', 'lpips'), batch_size=32, verbose=False, accelerator=None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    use_tracking = config['use_tracking']
    dataset = get_image_dataset(ds, root, mode=val_mode, image_size=image_size)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False)

    # image metric instance
    evaluator = ImageMetrics(metrics=metrics).to(device)

    results = {}
    ssims = []
    psnrs = []
    lpipss = []
    for i, batch in enumerate(tqdm(dataloader)):
        x = batch[0].to(device)
        if len(x.shape) == 5 and not use_tracking:
            # [bs, T, ch, h, w]
            x = x.view(-1, *x.shape[2:])
        elif len(x.shape) == 4 and use_tracking:
            # [bs, ch, h, w]
            x = x.unsqueeze(1)
        x_prior = x
        with torch.no_grad():
            output = model(x, x_prior=x_prior, deterministic=True)
            generated = output['rec'].clamp(0, 1)
            if len(x.shape) == 5:
                # [bs, T, ch, h, w]
                x = x.view(-1, *x.shape[2:])
            results = evaluator(x, generated)
        # [batch_size * T]
        if 'ssim' in metrics:
            ssims.append(results['ssim'])
        if 'psnr' in metrics:
            psnrs.append(results['psnr'])
        if 'lpips' in metrics:
            lpipss.append(results['lpips'])

    if 'ssim' in metrics:
        ssims = torch.cat(ssims, dim=0)
        mean_ssim = ssims.mean().data.cpu().item()
        std_ssim = ssims.std().data.cpu().item()
        results['ssim'] = mean_ssim
        results['ssim_std'] = std_ssim
    if 'psnr' in metrics:
        psnrs = torch.cat(psnrs, dim=0)
        mean_psnr = psnrs.mean().data.cpu().item()
        std_psnr = psnrs.std().data.cpu().item()
        results['psnr'] = mean_psnr
        results['psnr_std'] = std_psnr
    if 'lpips' in metrics:
        lpipss = torch.cat(lpipss, dim=0)
        mean_lpips = lpipss.mean().data.cpu().item()
        std_lpips = lpipss.std().data.cpu().item()
        results['lpips'] = mean_lpips
        results['lpips_std'] = std_lpips

    # save results
    path_to_conf = os.path.join(eval_dir, 'last_val_image_metrics.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(results, outfile, indent=2)

    del evaluator  # clear memory

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDLP Video Prediction Evaluation")
    parser.add_argument("-d", "--dataset", type=str, default='balls',
                        help="dataset to use: ['balls', 'traffic', 'clevrer', 'obj3d128', ...]")
    parser.add_argument("-p", "--path", type=str,
                        help="path to model directory, e.g. ./310822_141959_balls_ddlp")
    parser.add_argument("--checkpoint", type=str,
                        help="direct path to model checkpoint, e.g. ./checkpoints/ddlp-obj3d128/obj3d_ddlp.pth",
                        default="")
    parser.add_argument("--use_last", action='store_true',
                        help="use the last checkpoint instead of best")
    parser.add_argument("--use_train", action='store_true',
                        help="use the train set for the predictions")
    parser.add_argument("--sample", action='store_true',
                        help="use stochastic (non-deterministic) predictions")
    parser.add_argument("--cpu", action='store_true',
                        help="use cpu for inference")
    parser.add_argument("-c", "--cond_steps", type=int, help="the initial number of frames for predictions", default=-1)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=10)
    parser.add_argument("--horizon", type=int, help="timestep horizon for prediction", default=50)
    parser.add_argument("--prefix", type=str, default='',
                        help="prefix used for model saving")
    args = parser.parse_args()
    # parse input
    dir_path = args.path
    checkpoint_path = args.checkpoint
    ds = args.dataset
    use_train = args.use_train
    cond_steps = args.cond_steps
    timestep_horizon = args.horizon
    batch_size = args.batch_size
    use_cpu = args.cpu
    deterministic = not args.sample
    prefix = args.prefix
    # load model config
    model_ckpt_name = f'{ds}_ddlp{prefix}.pth'
    # model_best_ckpt_name = f'{ds}_ddlp{prefix}_best.pth'
    model_best_ckpt_name = f'{ds}_ddlp{prefix}_best_lpips.pth'
    use_last = args.use_last if os.path.exists(os.path.join(dir_path, f'saves/{model_best_ckpt_name}')) else True
    conf_path = os.path.join(dir_path, 'hparams.json')
    with open(conf_path, 'r') as f:
        config = json.load(f)
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    ds = config['ds']
    image_size = config['image_size']
    ch = config['ch']
    enc_channels = config['enc_channels']
    prior_channels = config['prior_channels']
    use_correlation_heatmaps = config['use_correlation_heatmaps']  # use heatmaps for tracking
    enable_enc_attn = config['enable_enc_attn']  # enable attention between patches in the particle encoder
    filtering_heuristic = config["filtering_heuristic"]  # filtering heuristic to filter prior keypoints

    model = ObjectDynamicsDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                              image_size=image_size, n_kp=config['n_kp'],
                              learned_feature_dim=config['learned_feature_dim'],
                              pad_mode=config['pad_mode'],
                              sigma=config['sigma'],
                              dropout=config['dropout'], patch_size=config['patch_size'],
                              n_kp_enc=config['n_kp_enc'],
                              n_kp_prior=config['n_kp_prior'], kp_range=config['kp_range'],
                              kp_activation=config['kp_activation'],
                              anchor_s=config['anchor_s'],
                              use_resblock=config['use_resblock'],
                              timestep_horizon=config['timestep_horizon'], predict_delta=config['predict_delta'],
                              scale_std=config['scale_std'],
                              offset_std=config['offset_std'], obj_on_alpha=config['obj_on_alpha'],
                              obj_on_beta=config['obj_on_beta'], pint_heads=config['pint_heads'],
                              pint_layers=config['pint_layers'], pint_dim=config['pint_dim'],
                              use_correlation_heatmaps=use_correlation_heatmaps,
                              enable_enc_attn=enable_enc_attn, filtering_heuristic=filtering_heuristic).to(device)
    if checkpoint_path.endswith('.pth'):
        ckpt_path = checkpoint_path
    else:
        ckpt_path = os.path.join(dir_path, f'saves/{model_ckpt_name if use_last else model_best_ckpt_name}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"loaded model from {ckpt_path}")

    # create dir for results
    pred_dir = os.path.join(dir_path, 'eval')
    os.makedirs(pred_dir, exist_ok=True)

    # conditional frames
    cond_steps = cond_steps if cond_steps > 0 else config['timestep_horizon']
    val_mode = 'train' if use_train else 'test'
    results = eval_ddlp_im_metric(model, device, timestep_horizon=timestep_horizon, val_mode=val_mode, config=config,
                                  eval_dir=pred_dir, cond_steps=cond_steps, metrics=('ssim', 'psnr', 'lpips'),
                                  batch_size=batch_size)
    print(f'results: {results}')
