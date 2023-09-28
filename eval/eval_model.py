"""
Evaluation of the ELBO on the validation set
"""
# imports
import numpy as np
import os
# torch
import torch
import torch.nn.functional as F
from utils.loss_functions import calc_reconstruction_loss, VGGDistance
from torch.utils.data import DataLoader
import torchvision.utils as vutils
# datasets
from datasets.get_dataset import get_video_dataset, get_image_dataset
# util functions
from utils.util_func import plot_keypoints_on_image_batch, animate_trajectories, \
    plot_bb_on_image_batch_from_z_scale_nms, plot_bb_on_image_batch_from_masks_nms


def evaluate_validation_elbo(model, config, epoch, batch_size=100, recon_loss_type="vgg", device=torch.device('cpu'),
                             save_image=False, fig_dir='./', topk=5, recon_loss_func=None, beta_rec=1.0, beta_kl=1.0,
                             kl_balance=0.001, accelerator=None, iou_thresh=0.2):
    model.eval()
    kp_range = model.kp_range
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    use_tracking = config['use_tracking']  # dataset root
    dataset = get_image_dataset(ds, root, mode='valid', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=False)
    if recon_loss_func is None:
        if recon_loss_type == "vgg":
            recon_loss_func = VGGDistance(device=device)
        else:
            recon_loss_func = calc_reconstruction_loss

    elbos = []
    for batch in dataloader:
        x = batch[0].to(device)
        if len(x.shape) == 5 and not use_tracking:
            # [bs, T, ch, h, w]
            x = x.view(-1, *x.shape[2:])
        elif len(x.shape) == 4 and use_tracking:
            # [bs, ch, h, w]
            x = x.unsqueeze(1)
        x_prior = x
        # forward pass
        with torch.no_grad():
            model_output = model(x, x_prior=x_prior)
            all_losses = model.calc_elbo(x, model_output, beta_kl=beta_kl,
                                         beta_rec=beta_rec, kl_balance=kl_balance,
                                         recon_loss_type=recon_loss_type,
                                         recon_loss_func=recon_loss_func)
        loss = all_losses['loss']

        mu_p = model_output['kp_p']
        mu = model_output['mu']
        z_base = model_output['z_base']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec']
        mu_scale = model_output['mu_scale']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        cropped_objects_original = model_output['cropped_objects_original']
        obj_on = model_output['obj_on']  # [batch_size, n_kp]
        alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]

        if use_tracking:
            x = x.view(-1, *x.shape[2:])
            x_prior = x_prior.view(-1, *x_prior.shape[2:])
        # for plotting, confidence calculation
        mu_tot = z_base + mu_offset
        logvar_tot = logvar_offset

        elbo = loss
        elbos.append(elbo.data.cpu().numpy())
    if save_image:
        max_imgs = 8
        mu_plot = mu_tot.clamp(min=kp_range[0], max=kp_range[1])
        img_with_kp = plot_keypoints_on_image_batch(mu_plot, x, radius=3,
                                                    thickness=1, max_imgs=max_imgs, kp_range=model.kp_range)
        img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x_prior, radius=3, thickness=1, max_imgs=max_imgs,
                                                      kp_range=model.kp_range)
        # top-k
        with torch.no_grad():
            logvar_sum = logvar_tot.sum(-1) * obj_on  # [bs, n_kp]
            logvar_topk = torch.topk(logvar_sum, k=topk, dim=-1, largest=False)
            indices = logvar_topk[1]  # [batch_size, topk]
            batch_indices = torch.arange(mu_tot.shape[0]).view(-1, 1).to(mu_tot.device)
            topk_kp = mu_tot[batch_indices, indices]
            # bounding boxes
            bb_scores = -1 * logvar_sum
            hard_threshold = None
            kp_batch = mu_plot
            scale_batch = mu_scale
            img_with_masks_nms, nms_ind = plot_bb_on_image_batch_from_z_scale_nms(kp_batch, scale_batch, x,
                                                                                  scores=bb_scores,
                                                                                  iou_thresh=iou_thresh,
                                                                                  thickness=1,
                                                                                  max_imgs=max_imgs,
                                                                                  hard_thresh=hard_threshold)
            alpha_masks = torch.where(alpha_masks < 0.05, 0.0, 1.0)
            img_with_masks_alpha_nms, _ = plot_bb_on_image_batch_from_masks_nms(alpha_masks, x,
                                                                                scores=bb_scores,
                                                                                iou_thresh=iou_thresh,
                                                                                thickness=1,
                                                                                max_imgs=max_imgs,
                                                                                hard_thresh=hard_threshold)
        img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                         radius=3, thickness=1, max_imgs=max_imgs,
                                                         kp_range=kp_range)
        dec_objects = model_output['dec_objects']
        bg = model_output['bg']
        if accelerator is not None:
            if accelerator.is_main_process:
                vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(accelerator.device),
                                             rec_x[:max_imgs, -3:],
                                             img_with_kp_p[:max_imgs, -3:].to(accelerator.device),
                                             img_with_kp_topk[:max_imgs, -3:].to(accelerator.device),
                                             dec_objects[:max_imgs, -3:],
                                             img_with_masks_nms[:max_imgs, -3:].to(accelerator.device),
                                             img_with_masks_alpha_nms[:max_imgs, -3:].to(accelerator.device),
                                             bg[:max_imgs, -3:]],
                                            dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                  nrow=8, pad_value=1)
            with torch.no_grad():
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.clone().reshape(-1, 3,
                                                                                    cropped_objects_original.shape[
                                                                                        -1],
                                                                                    cropped_objects_original.shape[
                                                                                        -1])
                if cropped_objects_original.shape[-1] != dec_objects_rgb.shape[-1]:
                    cropped_objects_original = F.interpolate(cropped_objects_original,
                                                             size=dec_objects_rgb.shape[-1],
                                                             align_corners=False, mode='bilinear')
            if accelerator.is_main_process:
                vutils.save_image(
                    torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                              dim=0).data.cpu(), '{}/image_obj_valid_{}.jpg'.format(fig_dir, epoch),
                    nrow=8, pad_value=1)
        else:
            vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(device),
                                         rec_x[:max_imgs, -3:],
                                         img_with_kp_p[:max_imgs, -3:].to(device),
                                         img_with_kp_topk[:max_imgs, -3:].to(device),
                                         dec_objects[:max_imgs, -3:],
                                         img_with_masks_nms[:max_imgs, -3:].to(device),
                                         img_with_masks_alpha_nms[:max_imgs, -3:].to(device),
                                         bg[:max_imgs, -3:]],
                                        dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                              nrow=8, pad_value=1)
            with torch.no_grad():
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.clone().reshape(-1, 3,
                                                                                    cropped_objects_original.shape[
                                                                                        -1],
                                                                                    cropped_objects_original.shape[
                                                                                        -1])
                if cropped_objects_original.shape[-1] != dec_objects_rgb.shape[-1]:
                    cropped_objects_original = F.interpolate(cropped_objects_original,
                                                             size=dec_objects_rgb.shape[-1],
                                                             align_corners=False, mode='bilinear')
            vutils.save_image(
                torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                          dim=0).data.cpu(), '{}/image_obj_valid_{}.jpg'.format(fig_dir, epoch),
                nrow=8, pad_value=1)
    return np.mean(elbos)


def evaluate_validation_elbo_dyn(model, config, epoch, batch_size=100, recon_loss_type="vgg",
                                 device=torch.device('cpu'),
                                 save_image=False, fig_dir='./', topk=5, recon_loss_func=None, beta_rec=1.0,
                                 beta_kl=1.0, beta_dyn=1.0, iou_thresh=0.2, beta_dyn_rec=1.0,
                                 kl_balance=1.0, accelerator=None, timestep_horizon=10,
                                 animation_horizon=50):
    model.eval()
    kp_range = model.kp_range
    # load data
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    cond_steps = config['cond_steps']  # dataset root
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon + 1, mode='valid', image_size=image_size)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=False)

    elbos = []
    for batch in dataloader:
        x = batch[0][:, :timestep_horizon + 1].to(device)
        x_prior = x
        with torch.no_grad():
            model_output = model(x, x_prior=x_prior)
            # calc elbo
            losses = model.calc_elbo(x, model_output, beta_kl=beta_kl,
                                     beta_dyn=beta_dyn, beta_rec=beta_rec, kl_balance=kl_balance,
                                     recon_loss_type=recon_loss_type, recon_loss_func=recon_loss_func,
                                     beta_dyn_rec=beta_dyn_rec)
        loss = losses['loss']

        mu_p = model_output['kp_p']
        mu = model_output['mu']
        z_base = model_output['z_base']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec']
        mu_scale = model_output['mu_scale']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        cropped_objects_original = model_output['cropped_objects_original']
        obj_on = model_output['obj_on']  # [batch_size, n_kp]
        alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]
        x = x.view(-1, *x.shape[2:])
        x_prior = x_prior.view(-1, *x_prior.shape[2:])
        # for plotting, confidence calculation
        mu_tot = z_base + mu_offset
        logvar_tot = logvar_offset

        elbo = loss
        elbos.append(elbo.data.cpu().numpy())
    if save_image:
        max_imgs = 8
        mu_plot = mu_tot.clamp(min=kp_range[0], max=kp_range[1])
        img_with_kp = plot_keypoints_on_image_batch(mu_plot, x, radius=3,
                                                    thickness=1, max_imgs=max_imgs, kp_range=model.kp_range)
        img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x_prior, radius=3, thickness=1, max_imgs=max_imgs,
                                                      kp_range=model.kp_range)
        # top-k
        with torch.no_grad():
            logvar_sum = logvar_tot.sum(-1) * obj_on  # [bs, n_kp]
            logvar_topk = torch.topk(logvar_sum, k=topk, dim=-1, largest=False)
            indices = logvar_topk[1]  # [batch_size, topk]
            batch_indices = torch.arange(mu_tot.shape[0]).view(-1, 1).to(mu_tot.device)
            topk_kp = mu_tot[batch_indices, indices]
            # bounding boxes
            bb_scores = -1 * logvar_sum
            hard_threshold = None
            kp_batch = mu_plot
            scale_batch = mu_scale
            img_with_masks_nms, nms_ind = plot_bb_on_image_batch_from_z_scale_nms(kp_batch, scale_batch, x,
                                                                                  scores=bb_scores,
                                                                                  iou_thresh=iou_thresh,
                                                                                  thickness=1,
                                                                                  max_imgs=max_imgs,
                                                                                  hard_thresh=hard_threshold)
            alpha_masks = torch.where(alpha_masks < 0.05, 0.0, 1.0)
            img_with_masks_alpha_nms, _ = plot_bb_on_image_batch_from_masks_nms(alpha_masks, x,
                                                                                scores=bb_scores,
                                                                                iou_thresh=iou_thresh,
                                                                                thickness=1,
                                                                                max_imgs=max_imgs,
                                                                                hard_thresh=hard_threshold)
        img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                         radius=3, thickness=1, max_imgs=max_imgs,
                                                         kp_range=kp_range)
        dec_objects = model_output['dec_objects']
        bg = model_output['bg']
        if accelerator is not None:
            if accelerator.is_main_process:
                vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(accelerator.device),
                                             rec_x[:max_imgs, -3:],
                                             img_with_kp_p[:max_imgs, -3:].to(accelerator.device),
                                             img_with_kp_topk[:max_imgs, -3:].to(accelerator.device),
                                             dec_objects[:max_imgs, -3:],
                                             img_with_masks_nms[:max_imgs, -3:].to(accelerator.device),
                                             img_with_masks_alpha_nms[:max_imgs, -3:].to(accelerator.device),
                                             bg[:max_imgs, -3:]],
                                            dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                  nrow=8, pad_value=1)
            with torch.no_grad():
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.clone().reshape(-1, 3,
                                                                                    cropped_objects_original.shape[
                                                                                        -1],
                                                                                    cropped_objects_original.shape[
                                                                                        -1])
                if cropped_objects_original.shape[-1] != dec_objects_rgb.shape[-1]:
                    cropped_objects_original = F.interpolate(cropped_objects_original,
                                                             size=dec_objects_rgb.shape[-1],
                                                             align_corners=False, mode='bilinear')
            if accelerator.is_main_process:
                vutils.save_image(
                    torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                              dim=0).data.cpu(), '{}/image_obj_valid_{}.jpg'.format(fig_dir, epoch),
                    nrow=8, pad_value=1)
        else:
            vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(device),
                                         rec_x[:max_imgs, -3:],
                                         img_with_kp_p[:max_imgs, -3:].to(device),
                                         img_with_kp_topk[:max_imgs, -3:].to(device),
                                         dec_objects[:max_imgs, -3:],
                                         img_with_masks_nms[:max_imgs, -3:].to(device),
                                         img_with_masks_alpha_nms[:max_imgs, -3:].to(device),
                                         bg[:max_imgs, -3:]],
                                        dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                              nrow=8, pad_value=1)
            with torch.no_grad():
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.clone().reshape(-1, 3,
                                                                                    cropped_objects_original.shape[
                                                                                        -1],
                                                                                    cropped_objects_original.shape[
                                                                                        -1])
                if cropped_objects_original.shape[-1] != dec_objects_rgb.shape[-1]:
                    cropped_objects_original = F.interpolate(cropped_objects_original,
                                                             size=dec_objects_rgb.shape[-1],
                                                             align_corners=False, mode='bilinear')
            vutils.save_image(
                torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                          dim=0).data.cpu(), '{}/image_obj_valid_{}.jpg'.format(fig_dir, epoch),
                nrow=8, pad_value=1)
        animate_trajectory_ddlp(model, config, epoch, device=device, fig_dir=fig_dir, prefix='valid_',
                                timestep_horizon=animation_horizon, num_trajetories=1,
                                accelerator=accelerator, train=False, cond_steps=cond_steps)
    return np.mean(elbos)


def animate_trajectory_ddlp(model, config, epoch, device=torch.device('cpu'), fig_dir='./', timestep_horizon=3,
                            num_trajetories=5, accelerator=None, train=False, prefix='', cond_steps=None,
                            deterministic=True):
    # load data
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    duration = config['animation_fps']

    mode = 'train' if train else "valid"
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon, mode=mode, image_size=image_size)

    batch_size = max(2, num_trajetories)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=False)
    batch = next(iter(dataloader))
    model_timestep_horizon = model.timestep_horizon
    cond_steps = model_timestep_horizon if cond_steps is None else cond_steps
    model.eval()
    x_horizon = batch[0][:, :timestep_horizon].to(device)
    # forward pass
    with torch.no_grad():
        preds = model.sample(x_horizon, num_steps=timestep_horizon - cond_steps, deterministic=deterministic,
                             bg_masks_from_fg=False, cond_steps=cond_steps)
        # preds: [bs, timestep_horizon, 3, im_size, im_size]
    for i in range(num_trajetories):
        gt_traj = x_horizon[i].permute(0, 2, 3, 1).data.cpu().numpy()
        pred_traj = preds[i].permute(0, 2, 3, 1).data.cpu().numpy()
        if accelerator is not None:
            if accelerator.is_main_process:
                animate_trajectories(gt_traj, pred_traj,
                                     path=os.path.join(fig_dir, f'{prefix}e{epoch}_traj_anim_{i}.gif'),
                                     duration=duration, rec_to_pred_t=cond_steps)
        else:
            animate_trajectories(gt_traj, pred_traj, path=os.path.join(fig_dir, f'{prefix}e{epoch}_traj_anim_{i}.gif'),
                                 duration=duration, rec_to_pred_t=cond_steps)
