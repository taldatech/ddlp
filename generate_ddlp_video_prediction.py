"""
Script to generate conditional video prediction from a pre-trained DDLP
"""
# imports
import os
import argparse
import json
# torch
import torch
# modules
from models import ObjectDynamicsDLP
# util functions
from eval.eval_model import animate_trajectory_ddlp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDLP Video Prediction")
    parser.add_argument("-d", "--dataset", type=str, default='balls',
                        help="dataset of to train the model on: ['traffic', 'clevrer', 'shapes']")
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
    parser.add_argument("--horizon", type=int, help="frame horizon to generate", default=50)
    parser.add_argument("-n", "--num_predictions", type=int, help="number of animations to generate", default=5)
    parser.add_argument("-c", "--cond_steps", type=int, help="the initial number of frames for predictions", default=-1)
    parser.add_argument("--prefix", type=str, default='',
                        help="prefix used for model saving")
    args = parser.parse_args()
    # parse input
    dir_path = args.path
    checkpoint_path = args.checkpoint
    ds = args.dataset
    use_train = args.use_train
    generation_horizon = args.horizon
    num_predictions = args.num_predictions
    cond_steps = args.cond_steps
    use_cpu = args.cpu
    deterministic = not args.sample
    prefix = args.prefix
    # load model config
    model_ckpt_name = f'{ds}_ddlp{prefix}.pth'
    model_best_ckpt_name = f'{ds}_ddlp{prefix}_best_lpips.pth'
    # model_best_ckpt_name = f'{ds}_ddlp{prefix}_best.pth'  # can also be used
    use_last = args.use_last if os.path.exists(os.path.join(dir_path, f'saves/{model_best_ckpt_name}')) else True
    conf_path = os.path.join(dir_path, 'hparams.json')
    with open(conf_path, 'r') as f:
        config = json.load(f)
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    image_size = config['image_size']
    ch = config['ch']
    enc_channels = config['enc_channels']
    prior_channels = config['prior_channels']
    use_correlation_heatmaps = config['use_correlation_heatmaps']
    enable_enc_attn = config['enable_enc_attn']
    filtering_heuristic = config['filtering_heuristic']

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
    model.requires_grad_(False)
    print(f"loaded model from {ckpt_path}")

    # create dir for videos
    pred_dir = os.path.join(dir_path, 'animations')
    os.makedirs(pred_dir, exist_ok=True)

    # conditional frames
    cond_steps = cond_steps if cond_steps > 0 else config['timestep_horizon']
    print(f'conditional input frames: {cond_steps}')
    print(f'deterministic predictions (use only mu): {deterministic}')
    # generate
    print('generating animations...')
    animate_trajectory_ddlp(model, config, epoch=0, device=device, fig_dir=pred_dir,
                            timestep_horizon=generation_horizon,
                            num_trajetories=num_predictions, accelerator=None, train=use_train, prefix='',
                            cond_steps=cond_steps, deterministic=deterministic)
