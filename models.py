"""
Main DLP and DDLP models.
"""
# imports
import numpy as np
# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
# modules
from modules.modules import KeyPointCNNOriginal, VariationalKeyPointPatchEncoder, CNNDecoder, \
    ObjectDecoderCNN, FCToCNN

from modules.modules import ParticleAttributeEncoder, ParticleFeaturesEncoder, ParticleFilterMixer
from modules.dynamics_modules import DynamicsDLP
# util functions
from utils.util_func import reparameterize, create_masks_fast, spatial_transform, calc_model_size
from utils.loss_functions import ChamferLossKL, calc_kl, calc_reconstruction_loss, VGGDistance, calc_kl_beta_dist


class FgDLP(nn.Module):
    def __init__(self, cdim=3, enc_channels=(16, 16, 32), prior_channels=(16, 16, 32), image_size=64, n_kp=1,
                 pad_mode='replicate', sigma=1.0, dropout=0.0,
                 patch_size=16, n_kp_enc=20, n_kp_prior=20, learned_feature_dim=16,
                 kp_range=(-1, 1), kp_activation="tanh", anchor_s=0.25,
                 use_resblock=False, use_correlation_heatmaps=True, enable_enc_attn=False,
                 filtering_heuristic='variance'):
        super(FgDLP, self).__init__()
        """
        DLP Foreground Module -- extract objects from an image
        cdim: channels of the input image (3...)
        enc_channels: channels for the posterior CNN (takes in the whole image)
        prior_channels: channels for prior CNN (takes in patches)
        n_kp: number of kp to extract from each (!) patch
        n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
        n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        sigma: the prior std of the KP
        dropout: dropout for the CNNs. We don't use it though...
        patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
        kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
        anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
        use_correlation_heatmaps: use correlation heatmaps as input to model particle properties (e.g., xy offset)
        enable_enc_attn: enable attention between patches features in the particle encoder
        filtering heuristic: filtering heuristic to filter prior keypoints,['distance', 'variance', 'random', 'none']
        """
        self.image_size = image_size
        self.sigma = sigma
        self.dropout = dropout
        self.kp_range = kp_range
        self.num_patches = int((image_size // patch_size) ** 2)
        self.n_kp = n_kp
        self.n_kp_total = self.n_kp * self.num_patches
        self.n_kp_prior = min(self.n_kp_total, n_kp_prior)
        self.n_kp_enc = n_kp_enc
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.anchor_patch_s = patch_size / image_size
        self.features_dim = int(image_size // (2 ** (len(enc_channels) - 1)))
        self.learned_feature_dim = learned_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.exclusive_patches = False
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.use_correlation_heatmaps = use_correlation_heatmaps
        self.enable_enc_attn = enable_enc_attn
        assert filtering_heuristic in ['distance', 'variance',
                                       'random', 'none'], f'unknown filtering heuristic: {filtering_heuristic}'
        self.filtering_heuristic = filtering_heuristic

        # prior
        self.prior = VariationalKeyPointPatchEncoder(cdim=cdim, channels=prior_channels, image_size=image_size,
                                                     n_kp=n_kp, kp_range=self.kp_range,
                                                     patch_size=patch_size,
                                                     pad_mode=pad_mode, sigma=sigma, dropout=dropout,
                                                     learnable_logvar=False, learned_feature_dim=0,
                                                     use_resblock=self.use_resblock)

        # posterior encoder
        if self.filtering_heuristic == 'none':
            # no filtering at all - we need a module that maps n_kp_prior -> n_kp_enc
            self.particle_mixer = ParticleFilterMixer(anchor_size=anchor_s, image_size=image_size,
                                                      margin=0, ch=cdim, n_kp_prior=self.n_kp_total,
                                                      n_kp_enc=self.n_kp_enc,
                                                      cnn_channels=prior_channels)
        else:
            self.particle_mixer = nn.Identity()
        # attribute encoder - anchor (z_a), offset (z_o), scale (z_s), transparency (z_t) and depth (z_d)
        self.particle_attribute_enc = ParticleAttributeEncoder(anchor_size=anchor_s, image_size=image_size,
                                                               margin=0, ch=cdim,
                                                               kp_activation=kp_activation,
                                                               use_resblock=self.use_resblock,
                                                               max_offset=1.0, cnn_channels=prior_channels,
                                                               use_correlation_heatmaps=use_correlation_heatmaps,
                                                               enable_attn=self.enable_enc_attn, attn_dropout=0.0)
        # appearance encoder - visual features encoder (z_f)
        self.particle_features_enc = ParticleFeaturesEncoder(anchor_s, learned_feature_dim,
                                                             image_size,
                                                             cnn_channels=prior_channels,
                                                             margin=0, enable_attn=self.enable_enc_attn,
                                                             attn_dropout=0.0)

        # object decoder
        self.object_dec = ObjectDecoderCNN(patch_size=(self.obj_patch_size, self.obj_patch_size), num_chans=4,
                                           bottleneck_size=learned_feature_dim, use_resblock=self.use_resblock)
        self.init_weights()

    def get_parameters(self, prior=True, encoder=True, decoder=True):
        parameters = []
        if prior:
            parameters.extend(list(self.prior.parameters()))
        if encoder:
            parameters.extend(list(self.particle_mixer.parameters()))
            parameters.extend(list(self.particle_attribute_enc.parameters()))
            # parameters.extend(list(self.particle_attribute_enc_dyn.parameters()))
            parameters.extend(list(self.particle_features_enc.parameters()))
        if decoder:
            parameters.extend(list(self.object_dec.parameters()))
        return parameters

    def set_require_grad(self, prior_value=True, enc_value=True, dec_value=True):
        # prior
        for param in self.prior.parameters():
            param.requires_grad = prior_value
        # encoder
        for param in self.particle_mixer.parameters():
            param.requires_grad = enc_value
        for param in self.particle_attribute_enc.parameters():
            param.requires_grad = enc_value
        # for param in self.particle_attribute_enc_dyn.parameters():
        #     param.requires_grad = enc_value
        for param in self.particle_features_enc.parameters():
            param.requires_grad = enc_value
        # decoder
        for param in self.object_dec.parameters():
            param.requires_grad = dec_value

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass

    def encode_all(self, x, deterministic=False, noisy=False, warmup=False, kp_init=None, cropped_objects_prev=None,
                   scale_prev=None, refinement_iter=False, with_offset=True):
        """
        2-stage encoding:
        0. if kp_init is None: create evenly spaced anchors. kp_init is z_base.
        1. attributes encoding: obj_on (z_t), depth (z_d), offset (z_o) and scale (z_s) encoding:
            produces [obj_on_a, obj_on_b] / [mu, logvar].
        2. features (z_f) encoding: [mu, logvar]
        """
        # kp_init: [batch_size, n_kp, 2] in [-1, 1]
        batch_size, ch, h, w = x.shape
        # 0. create or filter anchors
        if kp_init is None:
            # randomly sample n_kp_enc kp
            mu = torch.rand(batch_size, self.n_kp_enc, 2, device=x.device) * 2 - 1  # in [-1, 1]
        elif kp_init.shape[1] > self.n_kp_enc:
            mu = kp_init[:, :self.n_kp_enc]
        else:
            mu = kp_init
        logvar = torch.zeros_like(mu)
        z_base = mu + 0.0 * logvar  # deterministic value for chamfer-kl
        kp_heatmap = None  # backward compatibility, this is not used
        # 1. posterior offsets and scale, it is okay of scale_prev is None
        scale_in = None if (noisy or warmup) else scale_prev
        cropped_objects_prev = None if (noisy or warmup) else cropped_objects_prev
        particle_stats_dict = self.particle_attribute_enc(x, z_base.detach(),
                                                          previous_objects=cropped_objects_prev,
                                                          z_scale=scale_in)
        # first iteration encodes the refined anchor (z_a), then a second one to lock on target better (z_o)
        if refinement_iter:
            mu_offset = particle_stats_dict['mu']
            mu = z_base + mu_offset
            z_base = mu + 0.0 * logvar
            if scale_prev is None:
                scale_prev = None if (noisy or warmup) else particle_stats_dict['mu_scale'].detach()
            cropped_objects_prev = None if (noisy or warmup) else cropped_objects_prev
            particle_stats_dict = self.particle_attribute_enc(x, z_base.detach(),
                                                              previous_objects=cropped_objects_prev,
                                                              z_scale=scale_prev)

        if with_offset:
            mu_offset = particle_stats_dict['mu']
            logvar_offset = particle_stats_dict['logvar']
        else:
            mu_offset = torch.zeros_like(particle_stats_dict['mu'])
            logvar_offset = torch.zeros_like(particle_stats_dict['logvar'])
        mu_scale = particle_stats_dict['mu_scale']
        logvar_scale = particle_stats_dict['logvar_scale']

        lobj_on_a = particle_stats_dict['lobj_on_a']
        lobj_on_b = particle_stats_dict['lobj_on_b']
        mu_depth = particle_stats_dict['mu_depth']
        logvar_depth = particle_stats_dict['logvar_depth']
        # final position
        mu_tot = z_base + mu_offset
        logvar_tot = logvar_offset

        obj_on_a = lobj_on_a.exp().clamp_min(1e-5)
        obj_on_b = lobj_on_b.exp().clamp_min(1e-5)
        # if torch.isnan(obj_on_a).any():
        #     print(f'obj_on_a has nan')
        #     # torch.nan_to_num_(obj_on_a, nan=0.01)
        # if torch.isnan(obj_on_b).any():
        #     print(f'obj_on_b has nan')
        #     raise SystemError('NaNs detected')
        #     # torch.nan_to_num_(obj_on_b, nan=0.01)
        # try:
        #     obj_on_beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)
        # except ValueError:
        #     print(f'obj_on_a: {obj_on_a}')
        #     print(f'obj_on_b: {obj_on_b}')
        #     raise SystemError
        obj_on_beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)

        # reparameterize
        if deterministic:
            z = mu_tot
            z_offset = mu_offset
            z_scale = mu_scale
            z_depth = mu_depth
            z_obj_on = obj_on_beta_dist.mean
        else:
            z = reparameterize(mu_tot, logvar_tot) if with_offset else mu_tot
            z_offset = reparameterize(mu_offset, logvar_offset)  # not used
            z_scale = reparameterize(mu_scale, logvar_scale)
            z_depth = reparameterize(mu_depth, logvar_depth)
            z_obj_on = obj_on_beta_dist.rsample()

        # during warm-up and noisy stages we use small values around the patch size for the scale
        if z_scale is not None and noisy:
            anchor_size = self.anchor_s
            z_scale = 0.0 * z_scale + (np.log(anchor_size / (1 - anchor_size + 1e-5)) + 0.1 * torch.randn_like(z_scale))
        # to avoid null cases where obj_on -> 0, we noise its values during the noisy stage
        # z_obj_on = z_obj_on if not noisy else (0.0 * z_obj_on + 1.0)

        if warmup:
            z_base = z_base.detach()
            z = z.detach()
            z_scale = z_scale.detach()

        # 2. posterior attributes: obj_on, depth and visual features
        obj_enc_out = self.particle_features_enc(x, z, z_scale=z_scale.detach())

        mu_features = obj_enc_out['mu_features']
        logvar_features = obj_enc_out['logvar_features']
        cropped_objects = obj_enc_out['cropped_objects']

        # reparameterize
        if deterministic:
            z_features = mu_features
        else:
            z_features = reparameterize(mu_features, logvar_features)

        encode_dict = {'mu': mu, 'logvar': logvar, 'z_base': z_base, 'z': z, 'kp_heatmap': kp_heatmap,
                       'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'obj_on': z_obj_on,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'cropped_objects': cropped_objects,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
                       'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset}
        return encode_dict

    def encode_prior(self, x, x_prior=None, filtering_heuristic='variance', k=None):
        # encodes prior keypoints by patchifying the image and applying spatial-softmax
        if k is None:
            k = self.n_kp_prior
        if x_prior is None:
            x_prior = x
        kp_p, var_kp_p = self.prior(x_prior, global_kp=True)
        kp_p = kp_p.view(x_prior.shape[0], -1, 2)  # [batch_size, n_kp_total, 2]
        var_kp_p = var_kp_p.view(x_prior.shape[0], kp_p.shape[1], -1)  # [batch_size, n_kp_total, 3]
        if filtering_heuristic == 'distance':
            # filter proposals by distance to the patches' center
            dist_from_center = self.prior.get_distance_from_patch_centers(kp_p, global_kp=True)
            _, indices = torch.topk(dist_from_center, k=k, dim=-1, largest=True)
            batch_indices = torch.arange(kp_p.shape[0]).view(-1, 1).to(kp_p.device)
            kp_p = kp_p[batch_indices, indices]
        elif filtering_heuristic == 'variance':
            total_var = var_kp_p.sum(-1)
            _, indices = torch.topk(total_var, k=k, dim=-1, largest=False)
            batch_indices = torch.arange(kp_p.shape[0]).view(-1, 1).to(kp_p.device)
            kp_p = kp_p[batch_indices, indices]
        elif filtering_heuristic == 'none':
            return kp_p
        else:
            # alternatively, just sample random kp
            kp_p = kp_p[:, torch.randperm(kp_p.shape[1])[:k]]
        return kp_p

    def translate_patches(self, kp_batch, patches_batch, scale=None, translation=None, scale_normalized=False):
        """
        translate patches to be centered around given keypoints
        kp_batch: [bs, n_kp, 2] in [-1, 1]
        patches: [bs, n_kp, ch_patches, patch_size, patch_size]
        scale: None or [bs, n_kp, 2] or [bs, n_kp, 1]
        translation: None or [bs, n_kp, 2] or [bs, n_kp, 1] (delta from kp)
        scale_normalized: False if scale is not in [0, 1]
        :return: translated_padded_patches [bs, n_kp, ch, img_size, img_size]
        """
        batch_size, n_kp, ch_patch, patch_size, _ = patches_batch.shape
        img_size = self.image_size
        if scale is None:
            z_scale = (patch_size / img_size) * torch.ones_like(kp_batch)
        else:
            # normalize to [0, 1]
            if scale_normalized:
                z_scale = scale
            else:
                z_scale = torch.sigmoid(scale)  # -> [0, 1]
        z_pos = kp_batch.reshape(-1, kp_batch.shape[-1])  # [bs * n_kp, 2]
        z_scale = z_scale.view(-1, z_scale.shape[-1])  # [bs * n_kp, 2]
        patches_batch = patches_batch.reshape(-1, *patches_batch.shape[2:])
        out_dims = (batch_size * n_kp, ch_patch, img_size, img_size)
        trans_patches_batch = spatial_transform(patches_batch, z_pos, z_scale, out_dims, inverse=True)
        trans_padded_patches_batch = trans_patches_batch.view(batch_size, n_kp, *trans_patches_batch.shape[1:])
        # [bs, n_kp, ch, img_size, img_size]
        return trans_padded_patches_batch

    def get_objects_alpha_rgb(self, z_kp, z_features, z_scale=None, translation=None, noisy=False):
        # decode the latent particles into RGBA glimpses and place them on the canvas
        dec_objects = self.object_dec(z_features)  # [bs * n_kp, 4, patch_size, patch_size]
        dec_objects = dec_objects.view(-1, self.n_kp_enc,
                                       *dec_objects.shape[1:])  # [bs, n_kp, 4, patch_size, patch_size]
        # translate patches - place the decoded glimpses on the canvas
        dec_objects_trans = self.translate_patches(z_kp, dec_objects, z_scale, translation)
        dec_objects_trans = dec_objects_trans.clamp(0, 1)  # STN can change values to be < 0
        # dec_objects_trans: [bs, n_kp, 3, im_size, im_size]
        # multiply by alpha channel
        a_obj, rgb_obj = torch.split(dec_objects_trans, [1, 3], dim=2)

        if noisy:
            a_obj = a_obj + 0.1 * torch.randn_like(a_obj)
            a_obj = a_obj.clamp(0, 1)
        return dec_objects, a_obj, rgb_obj

    def get_objects_alpha_rgb_with_depth(self, a_obj, rgb_obj, obj_on, z_depth, eps=1e-5):
        # stitching the glimpses by factoring the alpha maps and the particle's inferred depth
        # obj_on: [bs, n_kp, 1]
        # z_depth: [bs, n_kp, 1]
        # turn off inactive particles
        a_obj = obj_on[:, :, None, None, None] * a_obj  # [bs, n_kp, 1, im_size, im_size]
        rgba_obj = a_obj * rgb_obj
        # importance map
        importance_map = a_obj * torch.sigmoid(-z_depth[:, :, :, None, None])
        # normalize
        importance_map = importance_map / (torch.sum(importance_map, dim=1, keepdim=True) + eps)
        # this imitates softmax to move objects on the depth axis
        dec_objects_trans = (rgba_obj * importance_map).sum(dim=1)
        alpha_mask = 1.0 - (importance_map * a_obj).sum(dim=1)
        a_obj = importance_map * a_obj
        return a_obj, alpha_mask, dec_objects_trans

    def decode_objects(self, z_kp, z_features, obj_on, z_scale=None, translation=None, noisy=False, z_depth=None):
        # stitching the decoded latent particles -> RGB, factoring the alpha maps and depths
        dec_objects, a_obj, rgb_obj = self.get_objects_alpha_rgb(z_kp, z_features, z_scale=z_scale,
                                                                 translation=translation, noisy=noisy)
        alpha_masks, bg_mask, dec_objects_trans = self.get_objects_alpha_rgb_with_depth(a_obj, rgb_obj, obj_on=obj_on,
                                                                                        z_depth=z_depth)
        return dec_objects, dec_objects_trans, alpha_masks, bg_mask

    def decode_all(self, z, z_features, obj_on, z_depth=None, noisy=False, z_scale=None):
        # a wrapper function to decode latent particles into and RGB image (no bg)
        object_dec_out = self.decode_objects(z, z_features, obj_on, noisy=noisy, z_depth=z_depth, z_scale=z_scale)
        dec_objects, dec_objects_trans, alpha_masks, bg_mask = object_dec_out

        decoder_out = {'dec_objects': dec_objects, 'dec_objects_trans': dec_objects_trans,
                       'bg_mask': bg_mask, 'alpha_masks': alpha_masks}

        return decoder_out

    def forward(self, x, deterministic=False, x_prior=None, warmup=False, noisy=False,
                cropped_objects_prev=None, mu_scale_prev=None, train_prior=True, refinement_iter=False):
        # refinement_iter: do another encoding step to get a better lock on the object's position (z_a + z_o)
        # first, extract prior KP proposals
        # prior proposals
        kp_p = self.encode_prior(x, x_prior=x_prior, filtering_heuristic=self.filtering_heuristic)
        kp_init = kp_p if train_prior else (0.0 * kp_p + kp_p.detach())  # 0.0 * kp_p is because of distributed training
        if self.filtering_heuristic == 'none':
            # n_kp_prior -> n_kp_enc
            kp_init = self.particle_mixer(x, kp_init)
        encoder_out = self.encode_all(x, deterministic=deterministic, noisy=noisy, warmup=warmup, kp_init=kp_init,
                                      cropped_objects_prev=cropped_objects_prev, scale_prev=mu_scale_prev,
                                      refinement_iter=refinement_iter)
        # detach for the kl-divergence
        kp_p = kp_p.detach()
        mu = encoder_out['mu']
        logvar = encoder_out['logvar']
        z_base = encoder_out['z_base']
        z = encoder_out['z']
        mu_offset = encoder_out['mu_offset']
        logvar_offset = encoder_out['logvar_offset']
        z_offset = encoder_out['z_offset']
        kp_heatmap = encoder_out['kp_heatmap']
        mu_features = encoder_out['mu_features']
        logvar_features = encoder_out['logvar_features']
        z_features = encoder_out['z_features']
        obj_on = encoder_out['obj_on']
        obj_on_a = encoder_out['obj_on_a']
        obj_on_b = encoder_out['obj_on_b']
        mu_depth = encoder_out['mu_depth']
        logvar_depth = encoder_out['logvar_depth']
        z_depth = encoder_out['z_depth']
        cropped_objects = encoder_out['cropped_objects']
        mu_scale = encoder_out['mu_scale']
        logvar_scale = encoder_out['logvar_scale']
        z_scale = encoder_out['z_scale']

        obj_on_sample = obj_on

        decoder_out = self.decode_all(z, z_features, obj_on_sample, z_depth, noisy=noisy, z_scale=z_scale)
        dec_objects = decoder_out['dec_objects']
        dec_objects_trans = decoder_out['dec_objects_trans']
        bg_mask = decoder_out['bg_mask']
        alpha_masks = decoder_out['alpha_masks']

        output_dict = {}
        output_dict['kp_p'] = kp_p
        output_dict['mu'] = mu
        output_dict['logvar'] = logvar
        output_dict['z_base'] = z_base
        output_dict['z'] = z
        output_dict['mu_offset'] = mu_offset
        output_dict['logvar_offset'] = logvar_offset
        output_dict['mu_features'] = mu_features
        output_dict['logvar_features'] = logvar_features
        output_dict['z_features'] = z_features
        output_dict['bg_mask'] = bg_mask
        output_dict['cropped_objects_original'] = cropped_objects
        output_dict['obj_on_a'] = obj_on_a
        output_dict['obj_on_b'] = obj_on_b
        output_dict['obj_on'] = obj_on
        output_dict['dec_objects_original'] = dec_objects
        output_dict['dec_objects'] = dec_objects_trans
        output_dict['mu_depth'] = mu_depth
        output_dict['logvar_depth'] = logvar_depth
        output_dict['z_depth'] = z_depth
        output_dict['mu_scale'] = mu_scale
        output_dict['logvar_scale'] = logvar_scale
        output_dict['z_scale'] = z_scale
        output_dict['alpha_masks'] = alpha_masks

        return output_dict


# class FgDDLP(nn.Module):
#     def __init__(self, cdim=3, enc_channels=(16, 16, 32), prior_channels=(16, 16, 32), image_size=64, n_kp=1,
#                  pad_mode='replicate', sigma=1.0, dropout=0.0,
#                  patch_size=16, n_kp_enc=20, n_kp_prior=20, learned_feature_dim=16,
#                  kp_range=(-1, 1), kp_activation="tanh", anchor_s=0.25,
#                  use_resblock=False, use_correlation_heatmaps=True, enable_enc_attn=False):
#         super(FgDDLP, self).__init__()
#         """
#         DDLP Foreground Module
#         Difference from DLP: separate encoders for static and dynamic statistics (e.g. mu/logvar)
#         cdim: channels of the input image (3...)
#         enc_channels: channels for the posterior CNN (takes in the whole image)
#         prior_channels: channels for prior CNN (takes in patches)
#         n_kp: number of kp to extract from each (!) patch
#         n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
#         n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
#         pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
#         sigma: the prior std of the KP
#         dropout: dropout for the CNNs. We don't use it though...
#         patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
#         kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
#         learned_feature_dim: the latent visual features dimensions extracted from glimpses.
#         kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
#         anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
#         use_correlation_heatmaps: use correlation heatmaps as input to model particle properties (e.g., xy offset)
#         enable_enc_attn: enable attention between patches features in the particle encoder
#         """
#         self.image_size = image_size
#         self.sigma = sigma
#         self.dropout = dropout
#         self.kp_range = kp_range
#         self.num_patches = int((image_size // patch_size) ** 2)
#         self.n_kp = n_kp
#         self.n_kp_total = self.n_kp * self.num_patches
#         self.n_kp_prior = min(self.n_kp_total, n_kp_prior)
#         self.n_kp_enc = n_kp_enc
#         self.kp_activation = kp_activation
#         self.patch_size = patch_size
#         self.anchor_patch_s = patch_size / image_size
#         self.features_dim = int(image_size // (2 ** (len(enc_channels) - 1)))
#         self.learned_feature_dim = learned_feature_dim
#         assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
#         self.anchor_s = anchor_s
#         self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
#         self.exclusive_patches = False
#         self.cdim = cdim
#         self.use_resblock = use_resblock
#         self.use_correlation_heatmaps = use_correlation_heatmaps
#         self.enable_enc_attn = enable_enc_attn
#
#         # prior
#         self.prior = VariationalKeyPointPatchEncoder(cdim=cdim, channels=prior_channels, image_size=image_size,
#                                                      n_kp=n_kp, kp_range=self.kp_range,
#                                                      patch_size=patch_size,
#                                                      pad_mode=pad_mode, sigma=sigma, dropout=dropout,
#                                                      learnable_logvar=False, learned_feature_dim=0,
#                                                      use_resblock=self.use_resblock)
#
#         # posterior encoder
#         self.particle_attribute_enc = ParticleAttributeEncoder(anchor_size=anchor_s, image_size=image_size,
#                                                                margin=0, ch=cdim,
#                                                                kp_activation=kp_activation,
#                                                                use_resblock=self.use_resblock,
#                                                                max_offset=1.0, cnn_channels=prior_channels,
#                                                                use_correlation_heatmaps=use_correlation_heatmaps,
#                                                                enable_attn=self.enable_enc_attn, attn_dropout=0.0)
#         self.particle_attribute_enc_dyn = ParticleAttributeEncoder(anchor_size=anchor_s, image_size=image_size,
#                                                                    margin=0, ch=cdim,
#                                                                    kp_activation=kp_activation,
#                                                                    use_resblock=self.use_resblock,
#                                                                    max_offset=1.0, cnn_channels=prior_channels,
#                                                                    use_correlation_heatmaps=use_correlation_heatmaps,
#                                                                    enable_attn=self.enable_enc_attn, attn_dropout=0.0)
#         self.particle_features_enc = ParticleFeaturesEncoder(anchor_s, learned_feature_dim,
#                                                              image_size,
#                                                              cnn_channels=prior_channels,
#                                                              margin=0, enable_attn=self.enable_enc_attn,
#                                                              attn_dropout=0.0)
#         self.particle_features_enc_dyn = ParticleFeaturesEncoder(anchor_s, learned_feature_dim,
#                                                                  image_size,
#                                                                  cnn_channels=prior_channels,
#                                                                  margin=0, enable_attn=self.enable_enc_attn,
#                                                                  attn_dropout=0.0)
#
#         # object decoder
#         self.object_dec = ObjectDecoderCNN(patch_size=(self.obj_patch_size, self.obj_patch_size), num_chans=4,
#                                            bottleneck_size=learned_feature_dim, use_resblock=self.use_resblock)
#         self.init_weights()
#
#     def get_parameters(self, prior=True, encoder=True, decoder=True):
#         parameters = []
#         if prior:
#             parameters.extend(list(self.prior.parameters()))
#         if encoder:
#             parameters.extend(list(self.particle_attribute_enc.parameters()))
#             parameters.extend(list(self.particle_features_enc.parameters()))
#             parameters.extend(list(self.particle_attribute_enc_dyn.parameters()))
#             parameters.extend(list(self.particle_features_enc_dyn.parameters()))
#         if decoder:
#             parameters.extend(list(self.object_dec.parameters()))
#         return parameters
#
#     def set_require_grad(self, prior_value=True, enc_value=True, dec_value=True):
#         # prior
#         for param in self.prior.parameters():
#             param.requires_grad = prior_value
#         # encoder
#         for param in self.particle_attribute_enc.parameters():
#             param.requires_grad = enc_value
#         for param in self.particle_attribute_enc_dyn.parameters():
#             param.requires_grad = enc_value
#         for param in self.particle_features_enc.parameters():
#             param.requires_grad = enc_value
#         for param in self.particle_features_enc_dyn.parameters():
#             param.requires_grad = enc_value
#         # decoder
#         for param in self.object_dec.parameters():
#             param.requires_grad = dec_value
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 # use pytorch's default
#                 pass
#
#     def encode_all(self, x, deterministic=False, noisy=False, warmup=False, kp_init=None, cropped_objects_prev=None,
#                    scale_prev=None, refinement_iter=False, use_static_encoders=True):
#         """
#         2-stage encoding:
#         0. if kp_init is None: create evenly spaced anchors. kp_init is z_base.
#         1. offset and scale encoding: produces offset and scale [mu, logvar].
#         2. attributes encoding: obj_on, depth and features [obj_on_a, obj_on_b] / [mu, logvar]
#         """
#         # kp_init: [batch_size, n_kp, 2] in [-1, 1]
#         batch_size, ch, h, w = x.shape
#         # 0. create or filter anchors
#         if kp_init is None:
#             # randomly sample n_kp_enc kp
#             mu = torch.rand(batch_size, self.n_kp_enc, 2, device=x.device) * 2 - 1  # in [-1, 1]
#         elif kp_init.shape[1] > self.n_kp_enc:
#             mu = kp_init[:, :self.n_kp_enc]
#         else:
#             mu = kp_init
#         logvar = torch.zeros_like(mu)
#         z_base = mu + 0.0 * logvar  # deterministic value for chamfer-kl
#         kp_heatmap = None  # backward compatibility, this is not used
#
#         attribute_enc = self.particle_attribute_enc if use_static_encoders else self.particle_attribute_enc_dyn
#         features_enc = self.particle_features_enc if use_static_encoders else self.particle_features_enc_dyn
#         # 1. posterior offsets and scale, it is okay of scale_prev is None
#         particle_stats_dict = attribute_enc(x, z_base.detach(), previous_objects=cropped_objects_prev,
#                                             z_scale=scale_prev)
#         # "second chance" to lock on target better
#         if refinement_iter:
#             mu_offset = particle_stats_dict['mu']
#             mu = z_base + mu_offset  # TODO: mu_offset.detach()?
#             z_base = mu + 0.0 * logvar
#             if scale_prev is None:
#                 scale_prev = particle_stats_dict['mu_scale']
#             particle_stats_dict = attribute_enc(x, z_base.detach(),
#                                                 previous_objects=cropped_objects_prev,
#                                                 z_scale=scale_prev.detach())
#         mu_offset = particle_stats_dict['mu']
#         logvar_offset = particle_stats_dict['logvar']
#         mu_scale = particle_stats_dict['mu_scale']
#         logvar_scale = particle_stats_dict['logvar_scale']
#
#         lobj_on_a = particle_stats_dict['lobj_on_a']
#         lobj_on_b = particle_stats_dict['lobj_on_b']
#         mu_depth = particle_stats_dict['mu_depth']
#         logvar_depth = particle_stats_dict['logvar_depth']
#         # final position
#         mu_tot = z_base + mu_offset
#         logvar_tot = logvar_offset
#
#         obj_on_a = lobj_on_a.exp().clamp_min(1e-5)
#         obj_on_b = lobj_on_b.exp().clamp_min(1e-5)
#         if torch.isnan(obj_on_a).any():
#             print(f'obj_on_a has nan')
#             torch.nan_to_num_(obj_on_a, nan=0.01)
#         if torch.isnan(obj_on_b).any():
#             print(f'obj_on_b has nan')
#             torch.nan_to_num_(obj_on_b, nan=0.01)
#         obj_on_beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)
#
#         # reparameterize
#         if deterministic:
#             z = mu_tot
#             z_offset = mu_offset
#             z_scale = mu_scale
#             z_depth = mu_depth
#             z_obj_on = obj_on_beta_dist.mean
#         else:
#             z = reparameterize(mu_tot, logvar_tot)
#             z_offset = reparameterize(mu_offset, logvar_offset)  # not used
#             z_scale = reparameterize(mu_scale, logvar_scale)
#             z_depth = reparameterize(mu_depth, logvar_depth)
#             z_obj_on = obj_on_beta_dist.rsample()
#
#         # during warm-up and noisy stages we use small values around the patch size for the scale
#         if z_scale is not None and noisy:
#             anchor_size = self.anchor_s
#             z_scale = 0.0 * z_scale + (np.log(anchor_size / (1 - anchor_size + 1e-5)) + 0.3 * torch.randn_like(z_scale))
#         # to avoid null cases where obj_on -> 0, we noise its values during the noisy stage
#         # z_obj_on = z_obj_on if not noisy else (z_obj_on + self.sigma * torch.randn_like(z_obj_on)).clamp(0, 1)
#
#         if warmup:
#             z_base = z_base.detach()
#             z = z.detach()
#             z_scale = z_scale.detach()
#
#         # 2. posterior attributes: obj_on, depth and visual features
#         obj_enc_out = features_enc(x, z, z_scale=z_scale.detach())
#
#         mu_features = obj_enc_out['mu_features']
#         logvar_features = obj_enc_out['logvar_features']
#         cropped_objects = obj_enc_out['cropped_objects']
#
#         # reparameterize
#         if deterministic:
#             z_features = mu_features
#         else:
#             z_features = reparameterize(mu_features, logvar_features)
#
#         encode_dict = {'mu': mu, 'logvar': logvar, 'z_base': z_base, 'z': z, 'kp_heatmap': kp_heatmap,
#                        'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
#                        'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'obj_on': z_obj_on,
#                        'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
#                        'cropped_objects': cropped_objects,
#                        'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
#                        'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset}
#         return encode_dict
#
#     def encode_prior(self, x, x_prior=None, filtering_heuristic='variance', k=None):
#         if k is None:
#             k = self.n_kp_prior
#         if x_prior is None:
#             x_prior = x
#         kp_p, var_kp_p = self.prior(x_prior, global_kp=True)
#         kp_p = kp_p.view(x_prior.shape[0], -1, 2)  # [batch_size, n_kp_total, 2]
#         var_kp_p = var_kp_p.view(x_prior.shape[0], kp_p.shape[1], -1)  # [batch_size, n_kp_total, 3]
#         if filtering_heuristic == 'distance':
#             # filter proposals by distance to the patches' center
#             dist_from_center = self.prior.get_distance_from_patch_centers(kp_p, global_kp=True)
#             _, indices = torch.topk(dist_from_center, k=k, dim=-1, largest=True)
#             batch_indices = torch.arange(kp_p.shape[0]).view(-1, 1).to(kp_p.device)
#             kp_p = kp_p[batch_indices, indices]
#         elif filtering_heuristic == 'variance':
#             total_var = var_kp_p.sum(-1)
#             _, indices = torch.topk(total_var, k=k, dim=-1, largest=False)
#             batch_indices = torch.arange(kp_p.shape[0]).view(-1, 1).to(kp_p.device)
#             kp_p = kp_p[batch_indices, indices]
#         else:
#             # alternatively, just sample random kp
#             kp_p = kp_p[:, torch.randperm(kp_p.shape[1])[:k]]
#         return kp_p
#
#     def translate_patches(self, kp_batch, patches_batch, scale=None, translation=None, scale_normalized=False):
#         """
#         translate patches to be centered around given keypoints
#         kp_batch: [bs, n_kp, 2] in [-1, 1]
#         patches: [bs, n_kp, ch_patches, patch_size, patch_size]
#         scale: None or [bs, n_kp, 2] or [bs, n_kp, 1]
#         translation: None or [bs, n_kp, 2] or [bs, n_kp, 1] (delta from kp)
#         scale_normalized: False if scale is not in [0, 1]
#         :return: translated_padded_pathces [bs, n_kp, ch, img_size, img_size]
#         """
#         batch_size, n_kp, ch_patch, patch_size, _ = patches_batch.shape
#         img_size = self.image_size
#         if scale is None:
#             z_scale = (patch_size / img_size) * torch.ones_like(kp_batch)
#         else:
#             # normalize to [0, 1]
#             if scale_normalized:
#                 z_scale = scale
#             else:
#                 z_scale = torch.sigmoid(scale)  # -> [0, 1]
#         z_pos = kp_batch.reshape(-1, kp_batch.shape[-1])  # [bs * n_kp, 2]
#         z_scale = z_scale.view(-1, z_scale.shape[-1])  # [bs * n_kp, 2]
#         patches_batch = patches_batch.reshape(-1, *patches_batch.shape[2:])
#         out_dims = (batch_size * n_kp, ch_patch, img_size, img_size)
#         trans_patches_batch = spatial_transform(patches_batch, z_pos, z_scale, out_dims, inverse=True)
#         trans_padded_patches_batch = trans_patches_batch.view(batch_size, n_kp, *trans_patches_batch.shape[1:])
#         # [bs, n_kp, ch, img_size, img_size]
#         return trans_padded_patches_batch
#
#     def get_objects_alpha_rgb(self, z_kp, z_features, z_scale=None, translation=None, noisy=False):
#         dec_objects = self.object_dec(z_features)  # [bs * n_kp, 4, patch_size, patch_size]
#         dec_objects = dec_objects.view(-1, self.n_kp_enc,
#                                        *dec_objects.shape[1:])  # [bs, n_kp, 4, patch_size, patch_size]
#         # translate patches
#         dec_objects_trans = self.translate_patches(z_kp, dec_objects, z_scale, translation)
#         dec_objects_trans = dec_objects_trans.clamp(0, 1)  # STN can change values to be < 0
#         # dec_objects_trans: [bs, n_kp, 3, im_size, im_size]
#         # multiply by alpha channel
#         a_obj, rgb_obj = torch.split(dec_objects_trans, [1, 3], dim=2)
#
#         if noisy:
#             a_obj = a_obj + 0.1 * torch.randn_like(a_obj)
#             a_obj = a_obj.clamp(0, 1)
#         return dec_objects, a_obj, rgb_obj
#
#     def get_objects_alpha_rgb_with_depth(self, a_obj, rgb_obj, obj_on, z_depth, eps=1e-5):
#         # obj_on: [bs, n_kp, 1]
#         # z_depth: [bs, n_kp, 1]
#         # turn off inactive particles
#         a_obj = obj_on[:, :, None, None, None] * a_obj  # [bs, n_kp, 1, im_size, im_size]
#         rgba_obj = a_obj * rgb_obj
#         # importance map
#         importance_map = a_obj * torch.sigmoid(-z_depth[:, :, :, None, None])
#         # normalize
#         importance_map = importance_map / (torch.sum(importance_map, dim=1, keepdim=True) + eps)
#         # this imitates softmax
#         dec_objects_trans = (rgba_obj * importance_map).sum(dim=1)
#         alpha_mask = 1.0 - (importance_map * a_obj).sum(dim=1)
#         a_obj = importance_map * a_obj
#         return a_obj, alpha_mask, dec_objects_trans
#
#     def decode_objects(self, z_kp, z_features, obj_on, z_scale=None, translation=None, noisy=False, z_depth=None):
#         dec_objects, a_obj, rgb_obj = self.get_objects_alpha_rgb(z_kp, z_features, z_scale=z_scale,
#                                                                  translation=translation, noisy=noisy)
#         alpha_masks, bg_mask, dec_objects_trans = self.get_objects_alpha_rgb_with_depth(a_obj, rgb_obj, obj_on=obj_on,
#                                                                                         z_depth=z_depth)
#         return dec_objects, dec_objects_trans, alpha_masks, bg_mask
#
#     def decode_all(self, z, z_features, obj_on, z_depth=None, noisy=False, z_scale=None):
#         object_dec_out = self.decode_objects(z, z_features, obj_on, noisy=noisy, z_depth=z_depth, z_scale=z_scale)
#         dec_objects, dec_objects_trans, alpha_masks, bg_mask = object_dec_out
#
#         decoder_out = {'dec_objects': dec_objects, 'dec_objects_trans': dec_objects_trans,
#                        'bg_mask': bg_mask, 'alpha_masks': alpha_masks}
#
#         return decoder_out
#
#     def forward(self, x, deterministic=False, x_prior=None, warmup=False, noisy=False,
#                 cropped_objects_prev=None, mu_scale_prev=None, train_prior=True, refinement_iter=False,
#                 use_static_encoders=True):
#         # refinement_iter: do another encoding step to get a better lock on the object's position
#         # first, extract prior KP proposals
#         # prior proposals
#         kp_p = self.encode_prior(x, x_prior=x_prior, filtering_heuristic='variance')
#         kp_init = kp_p if train_prior else kp_p.detach()
#         encoder_out = self.encode_all(x, deterministic=deterministic, noisy=noisy, warmup=warmup, kp_init=kp_init,
#                                       cropped_objects_prev=cropped_objects_prev, scale_prev=mu_scale_prev,
#                                       refinement_iter=refinement_iter, use_static_encoders=use_static_encoders)
#         # detach for the kl-divergence
#         kp_p = kp_p.detach()
#         mu = encoder_out['mu']
#         logvar = encoder_out['logvar']
#         z_base = encoder_out['z_base']
#         z = encoder_out['z']
#         mu_offset = encoder_out['mu_offset']
#         logvar_offset = encoder_out['logvar_offset']
#         z_offset = encoder_out['z_offset']
#         kp_heatmap = encoder_out['kp_heatmap']
#         mu_features = encoder_out['mu_features']
#         logvar_features = encoder_out['logvar_features']
#         z_features = encoder_out['z_features']
#         obj_on = encoder_out['obj_on']
#         obj_on_a = encoder_out['obj_on_a']
#         obj_on_b = encoder_out['obj_on_b']
#         mu_depth = encoder_out['mu_depth']
#         logvar_depth = encoder_out['logvar_depth']
#         z_depth = encoder_out['z_depth']
#         cropped_objects = encoder_out['cropped_objects']
#         mu_scale = encoder_out['mu_scale']
#         logvar_scale = encoder_out['logvar_scale']
#         z_scale = encoder_out['z_scale']
#
#         obj_on_sample = obj_on
#
#         decoder_out = self.decode_all(z, z_features, obj_on_sample, z_depth, noisy=noisy, z_scale=z_scale)
#         dec_objects = decoder_out['dec_objects']
#         dec_objects_trans = decoder_out['dec_objects_trans']
#         bg_mask = decoder_out['bg_mask']
#         alpha_masks = decoder_out['alpha_masks']
#
#         output_dict = {}
#         output_dict['kp_p'] = kp_p
#         output_dict['mu'] = mu
#         output_dict['logvar'] = logvar
#         output_dict['z_base'] = z_base
#         output_dict['z'] = z
#         output_dict['mu_offset'] = mu_offset
#         output_dict['logvar_offset'] = logvar_offset
#         output_dict['mu_features'] = mu_features
#         output_dict['logvar_features'] = logvar_features
#         output_dict['z_features'] = z_features
#         output_dict['bg_mask'] = bg_mask
#         output_dict['cropped_objects_original'] = cropped_objects
#         output_dict['obj_on_a'] = obj_on_a
#         output_dict['obj_on_b'] = obj_on_b
#         output_dict['obj_on'] = obj_on
#         output_dict['dec_objects_original'] = dec_objects
#         output_dict['dec_objects'] = dec_objects_trans
#         output_dict['mu_depth'] = mu_depth
#         output_dict['logvar_depth'] = logvar_depth
#         output_dict['z_depth'] = z_depth
#         output_dict['mu_scale'] = mu_scale
#         output_dict['logvar_scale'] = logvar_scale
#         output_dict['z_scale'] = z_scale
#         output_dict['alpha_masks'] = alpha_masks
#
#         return output_dict


class BgDLP(nn.Module):
    def __init__(self, cdim=3, enc_channels=(16, 16, 32), image_size=64, pad_mode='replicate', dropout=0.0,
                 learned_feature_dim=16, n_kp_enc=10, use_resblock=False):
        super(BgDLP, self).__init__()
        """
        DLP Background Module -- encode a latent for the (masked) background, z_bg
        Basically, just a convolutional-based encoder used in standard VAEs
        cdim: channels of the input image (3...)
        enc_channels: channels for the posterior CNN (takes in the whole image)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        """
        self.image_size = image_size
        self.dropout = dropout
        self.features_dim = int(image_size // (2 ** (len(enc_channels) - 1)))
        self.learned_feature_dim = learned_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.cdim = cdim
        self.n_kp_enc = 32
        self.use_resblock = use_resblock

        # encoder
        self.bg_cnn_enc = KeyPointCNNOriginal(cdim=cdim, channels=enc_channels, image_size=image_size,
                                              n_kp=self.n_kp_enc,
                                              pad_mode=pad_mode, use_resblock=self.use_resblock)
        bg_enc_output_dim = self.learned_feature_dim * 2  # [mu_features, sigma_features]
        self.bg_enc = nn.Sequential(nn.Linear(self.n_kp_enc * self.features_dim ** 2, 256),
                                    nn.ReLU(True),
                                    nn.Linear(256, 256),
                                    nn.ReLU(True),
                                    nn.Linear(256, bg_enc_output_dim))
        # decoder
        decoder_n_kp = self.n_kp_enc
        self.latent_to_feat_map = FCToCNN(target_hw=self.features_dim, n_ch=decoder_n_kp,
                                          features_dim=self.learned_feature_dim, pad_mode=pad_mode,
                                          use_resblock=self.use_resblock)
        self.dec = CNNDecoder(cdim=cdim, channels=enc_channels, image_size=image_size, in_ch=decoder_n_kp,
                              pad_mode=pad_mode, use_resblock=self.use_resblock)
        self.init_weights()

    def get_parameters(self, prior=True, encoder=True, decoder=True):
        parameters = []
        if encoder:
            parameters.extend(list(self.bg_cnn_enc.parameters()))
            parameters.extend(list(self.bg_enc.parameters()))
        if decoder:
            parameters.extend(list(self.dec.parameters()))
            parameters.extend(list(self.latent_to_feat_map.parameters()))
        return parameters

    def set_require_grad(self, prior_value=True, enc_value=True, dec_value=True):
        for param in self.bg_cnn_enc.parameters():
            param.requires_grad = enc_value
        for param in self.bg_enc.parameters():
            param.requires_grad = enc_value
        for param in self.dec.parameters():
            param.requires_grad = dec_value
        for param in self.latent_to_feat_map.parameters():
            param.requires_grad = dec_value

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass

    def encode_bg_features(self, x, masks=None):
        # x: [bs, ch, image_size, image_size]
        # masks: [bs, 1, image_size, image_size]
        batch_size, _, features_dim, _ = x.shape
        # bg features
        if masks is not None:
            x_in = x * masks
        else:
            x_in = x
        _, cnn_features = self.bg_cnn_enc(x_in)
        cnn_features = cnn_features.view(batch_size, -1)  # flatten
        bg_enc_out = self.bg_enc(cnn_features)  # [bs,, 2 * learned_features_dim]
        mu_bg, logvar_bg = bg_enc_out.chunk(2, dim=-1)

        return mu_bg, logvar_bg

    def encode_all(self, x, masks=None, deterministic=False):
        # encode background
        mu_bg, logvar_bg = self.encode_bg_features(x, masks)
        if deterministic:
            z_bg = mu_bg
        else:
            z_bg = reparameterize(mu_bg, logvar_bg)
        z_kp = torch.zeros(mu_bg.shape[0], 1, 2, device=x.device, dtype=torch.float)
        encode_dict = {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp}
        return encode_dict

    def decode_all(self, z_features):
        feature_maps = self.latent_to_feat_map(z_features)
        bg_rec = self.dec(feature_maps)
        return bg_rec

    def forward(self, x, masks=None, deterministic=False):
        encoder_out = self.encode_all(x, masks, deterministic)
        mu_bg = encoder_out['mu_bg']
        logvar_bg = encoder_out['logvar_bg']
        z_bg = encoder_out['z_bg']
        z_kp = encoder_out['z_kp']
        bg_rec = self.decode_all(z_bg)
        output_dict = {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp, 'bg_rec': bg_rec}
        return output_dict


# single-image deep latent particles
class ObjectDLP(nn.Module):
    def __init__(self, cdim=3, enc_channels=(16, 16, 32), prior_channels=(16, 16, 32), image_size=64, n_kp=1,
                 pad_mode='replicate', sigma=1.0, dropout=0.0,
                 patch_size=16, n_kp_enc=20, n_kp_prior=20, learned_feature_dim=16,
                 bg_learned_feature_dim=None,
                 kp_range=(-1, 1), kp_activation="tanh", anchor_s=0.25, use_tracking=False,
                 use_resblock=False, scale_std=0.3, offset_std=0.2, obj_on_alpha=0.1, obj_on_beta=0.1,
                 use_correlation_heatmaps=False, enable_enc_attn=False, filtering_heuristic='variance'):
        super(ObjectDLP, self).__init__()
        """
        cdim: channels of the input image (3...)
        enc_channels: channels for the posterior CNN (takes in the whole image)
        prior_channels: channels for prior CNN (takes in patches)
        n_kp: number of kp to extract from each (!) patch
        n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
        n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        sigma: the prior std of the KP
        dropout: dropout for the CNNs. We don't use it though...
        patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
        kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        bg_learned_feature_dim: the latent visual features dimensions extracted from masked bg. None-> =learned_feature_dim
        kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
        anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
        scale_std: prior std for the scale
        offset_std: prior std for the offset
        obj_on_alpha: prior alpha (Beta distribution) for obj_on
        obj_on_beta: prior beta (Beta distribution) for obj_on
        enable_enc_attn: enable attention between patches in the particle encoder
        use_tracking: use tracking mechanism
        use_correlation_heatmaps: use correlation heatmaps for tracking
        filtering heuristic: filtering heuristic to filter prior keypoints,['distance', 'variance', 'random', 'none']
        """
        self.image_size = image_size
        self.sigma = sigma
        self.dropout = dropout
        self.kp_range = kp_range
        self.num_patches = int((image_size // patch_size) ** 2)
        self.n_kp = n_kp
        self.n_kp_total = self.n_kp * self.num_patches
        self.n_kp_prior = min(self.n_kp_total, n_kp_prior)
        self.n_kp_enc = n_kp_enc
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.features_dim = int(image_size // (2 ** (len(enc_channels) - 1)))
        self.learned_feature_dim = learned_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.bg_learned_feature_dim = learned_feature_dim if bg_learned_feature_dim is None else bg_learned_feature_dim
        assert self.bg_learned_feature_dim > 0, "bg_learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * image_size).astype(int)
        # print(f'object patch size: {self.obj_patch_size}')
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.use_tracking = use_tracking
        self.use_correlation_heatmaps = use_correlation_heatmaps and self.use_tracking
        self.enable_enc_attn = enable_enc_attn
        assert filtering_heuristic in ['distance', 'variance',
                                       'random', 'none'], f'unknown filtering heuristic: {filtering_heuristic}'
        self.filtering_heuristic = filtering_heuristic

        # priors
        self.register_buffer('logvar_kp', torch.log(torch.tensor(sigma ** 2)))
        self.register_buffer('mu_scale_prior',
                             torch.tensor(np.log(0.75 * self.anchor_s / (1 - 0.75 * self.anchor_s + 1e-5))))
        self.register_buffer('logvar_scale_p', torch.log(torch.tensor(scale_std ** 2)))
        self.register_buffer('logvar_offset_p', torch.log(torch.tensor(offset_std ** 2)))
        self.register_buffer('obj_on_a_p', torch.tensor(obj_on_alpha))
        self.register_buffer('obj_on_b_p', torch.tensor(obj_on_beta))

        # foreground module
        self.fg_module = FgDLP(cdim=cdim, enc_channels=enc_channels, prior_channels=prior_channels,
                               image_size=image_size, n_kp=n_kp, pad_mode=pad_mode,
                               sigma=sigma, dropout=dropout, patch_size=patch_size, n_kp_enc=n_kp_enc,
                               n_kp_prior=n_kp_prior, learned_feature_dim=learned_feature_dim, kp_range=kp_range,
                               kp_activation=kp_activation, anchor_s=anchor_s,
                               use_resblock=self.use_resblock,
                               use_correlation_heatmaps=use_correlation_heatmaps, enable_enc_attn=enable_enc_attn,
                               filtering_heuristic=filtering_heuristic)

        # background module
        self.bg_module = BgDLP(cdim=cdim, enc_channels=enc_channels, image_size=image_size, pad_mode=pad_mode,
                               dropout=dropout, learned_feature_dim=self.bg_learned_feature_dim, n_kp_enc=n_kp_enc,
                               use_resblock=self.use_resblock)
        self.init_weights()

    def get_parameters(self, prior=True, encoder=True, decoder=True):
        parameters = []
        parameters.extend(self.fg_module.get_parameters(prior, encoder, decoder))
        parameters.extend(self.bg_module.get_parameters(prior, encoder, decoder))
        return parameters

    def set_require_grad(self, prior_value=True, enc_value=True, dec_value=True):
        self.fg_module.set_require_grad(prior_value, enc_value, dec_value)
        self.bg_module.set_require_grad(prior_value, enc_value, dec_value)

    def init_weights(self):
        self.fg_module.init_weights()
        self.bg_module.init_weights()

    def info(self):
        log_str = f'prior (patch) kp filtering: {self.n_kp_total} -> prior kp: {self.n_kp_prior}\n'
        log_str += f'prior (patch) kp filtering method: {self.filtering_heuristic}\n'
        log_str += f'prior patch size: {self.patch_size}\n'
        log_str += f'# posterior particles: {self.n_kp_enc}\n'
        log_str += f'particle visual feature dim: {self.learned_feature_dim}\n'
        log_str += f'bg visual feature dim: {self.bg_learned_feature_dim}\n'
        log_str += f'posterior object patch size: {self.obj_patch_size}\n'
        # cnns output sizes
        # prior
        prior_cnn_size = self.fg_module.prior.enc.conv_output_size
        # posterior - objects
        fg_cnn_size = self.fg_module.particle_attribute_enc.cnn.conv_output_size
        # posterior - bg
        bg_cnn_size = self.bg_module.bg_cnn_enc.conv_output_size
        log_str += f'cnn pre-pool out sizes: prior-{prior_cnn_size}, obj enc-{fg_cnn_size}, bg enc-{bg_cnn_size}\n'
        # object decoder
        dec_upsample_l = int(np.log(self.fg_module.object_dec.patch_size[0]) // np.log(2)) - 3
        log_str += f'object decoder # upsampling layers: {dec_upsample_l}\n'
        # tracking
        log_str += f'tracking: {self.use_tracking}, correlation maps for tracking: {self.use_correlation_heatmaps}\n'
        # num parameters and model size
        size_dict = calc_model_size(self)
        size_mb = size_dict['size_mb']
        n_params = size_dict['n_params']
        log_str += f'trainable parameters: {n_params} ({n_params / (10 ** 6):.4f}M)\n'
        log_str += f'estimated size on disk: {size_mb:.3f}MB\n'
        return log_str

    def get_bg_mask_from_particle_glimpses(self, z, z_obj_on, mask_size):
        """
        generates a mask based on particles position and the scale. Masks are squares.
        """
        obj_fmap_masks = create_masks_fast(z.detach(), anchor_s=self.anchor_s, feature_dim=mask_size)
        obj_fmap_masks = obj_fmap_masks.clamp(0, 1) * z_obj_on[:, :, None, None, None].detach()
        bg_mask = 1 - obj_fmap_masks.squeeze(2).sum(1, keepdim=True).clamp(0, 1)
        return bg_mask

    def fg_sequential_opt(self, x, deterministic=False, x_prior=None, warmup=False, noisy=False, reshape=True,
                          train_prior=False, decode=True):
        """
        sequential encoding per-timestep, for tracking purposes
        reshape: whether to reshape to [bs * timestep_horizon, ...]
        train_prior: should the prior (proposal) keypoints have gradients
        num_static_frames: the number of frames that are optimized w.r.t the constant prior and not the dynamics
        """
        # x: [bs, T + 1, ...]
        batch_size, timestep_horizon = x.size(0), x.size(1)
        num_static_frames = timestep_horizon
        if x_prior is None:
            # should be None in the single-image setting
            x_prior = x
        # for the first time step, standard encoding
        filtering_heuristic = 'random' if (warmup or noisy) else self.filtering_heuristic
        kp_p = self.fg_module.encode_prior(x[:, :num_static_frames].reshape(-1, *x.shape[2:]),
                                           x_prior=x_prior[:, :num_static_frames].reshape(-1, *x_prior.shape[2:]),
                                           filtering_heuristic=filtering_heuristic)
        kp_p = kp_p.reshape(batch_size, num_static_frames, *kp_p.shape[1:])  # [bs, n_stat_frames, n_kp_p, 2]
        kp_p = kp_p if train_prior else kp_p.detach()
        kp_init = kp_p[:, 0]  # first timestep
        if self.filtering_heuristic == 'none':
            # n_kp_prior -> n_kp_enc
            kp_init = self.particle_mixer(x[:, 0], kp_init)
        fg_dict = self.fg_module.encode_all(x[:, 0], deterministic=deterministic, warmup=warmup, noisy=noisy,
                                            kp_init=kp_init, refinement_iter=True)
        kp_p = kp_p.detach()  # freeze w.r.t to kl-divergence
        # kp_p = fg_dict['kp_p']
        mu = fg_dict['mu']
        logvar = fg_dict['logvar']
        z_base = fg_dict['z_base']
        z = fg_dict['z']
        mu_offset = fg_dict['mu_offset']
        logvar_offset = fg_dict['logvar_offset']
        mu_features = fg_dict['mu_features']
        logvar_features = fg_dict['logvar_features']
        z_features = fg_dict['z_features']
        cropped_objects = fg_dict['cropped_objects']
        obj_on_a = fg_dict['obj_on_a']
        obj_on_b = fg_dict['obj_on_b']
        z_obj_on = fg_dict['obj_on']
        mu_depth = fg_dict['mu_depth']
        logvar_depth = fg_dict['logvar_depth']
        z_depth = fg_dict['z_depth']
        mu_scale = fg_dict['mu_scale']
        logvar_scale = fg_dict['logvar_scale']
        z_scale = fg_dict['z_scale']

        # initialize lists to collect all outputs
        mus, logvars, zs, z_bases = [mu], [logvar], [z], [z_base]
        mu_offsets, logvar_offsets = [mu_offset], [logvar_offset]
        mu_featuress, logvar_featuress, z_featuress = [mu_features], [logvar_features], [z_features]
        cropped_objectss = [cropped_objects]
        obj_on_as, obj_on_bs, z_obj_ons = [obj_on_a], [obj_on_b], [z_obj_on]
        mu_depths, logvar_depths, z_depths = [mu_depth], [logvar_depth], [z_depth]
        mu_scales, logvar_scales, z_scales = [mu_scale], [logvar_scale], [z_scale]

        for i in range(1, timestep_horizon):
            # tracking, search for mu_tot in the area of the previous mu
            mu_prev = zs[-1].detach()
            cropped_objects_prev = cropped_objectss[-1].detach()
            cropped_objects_prev = cropped_objects_prev.view(-1, *cropped_objects_prev.shape[2:])
            mu_scale_prev = z_scales[-1].detach()
            fg_dict = self.fg_module.encode_all(x[:, i], deterministic=deterministic, warmup=warmup,
                                                noisy=noisy, kp_init=mu_prev,
                                                cropped_objects_prev=cropped_objects_prev, scale_prev=mu_scale_prev)
            mu = fg_dict['mu']
            logvar = fg_dict['logvar']
            z_base = fg_dict['z_base']
            z = fg_dict['z']
            mu_offset = fg_dict['mu_offset']
            logvar_offset = fg_dict['logvar_offset']
            mu_features = fg_dict['mu_features']
            logvar_features = fg_dict['logvar_features']
            z_features = fg_dict['z_features']
            cropped_objects = fg_dict['cropped_objects']
            obj_on_a = fg_dict['obj_on_a']
            obj_on_b = fg_dict['obj_on_b']
            z_obj_on = fg_dict['obj_on']
            mu_depth = fg_dict['mu_depth']
            logvar_depth = fg_dict['logvar_depth']
            z_depth = fg_dict['z_depth']
            mu_scale = fg_dict['mu_scale']
            logvar_scale = fg_dict['logvar_scale']
            z_scale = fg_dict['z_scale']

            mus.append(mu)
            logvars.append(logvar)
            z_bases.append(z_base)
            zs.append(z)
            mu_offsets.append(mu_offset)
            logvar_offsets.append(logvar_offset)
            mu_featuress.append(mu_features)
            logvar_featuress.append(logvar_features)
            z_featuress.append(z_features)
            cropped_objectss.append(cropped_objects)
            obj_on_as.append(obj_on_a)
            obj_on_bs.append(obj_on_b)
            z_obj_ons.append(z_obj_on)
            mu_depths.append(mu_depth)
            logvar_depths.append(logvar_depth)
            z_depths.append(z_depth)
            mu_scales.append(mu_scale)
            logvar_scales.append(logvar_scale)
            z_scales.append(z_scale)

        # pad the kp proposals (prior) tensor, only care about t=[0, num_static_frames]
        num_pad = len(mus) - kp_p.shape[1]
        if num_pad > 0:
            kp_p_pad = kp_p[:, -1:].detach()
            kp_p = torch.cat([kp_p, kp_p_pad.repeat(1, num_pad, 1, 1)], dim=1)
        kp_ps = kp_p
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)
        z_bases = torch.stack(z_bases, dim=1)
        zs = torch.stack(zs, dim=1)
        mu_offsets = torch.stack(mu_offsets, dim=1)
        logvar_offsets = torch.stack(logvar_offsets, dim=1)
        mu_featuress = torch.stack(mu_featuress, dim=1)
        logvar_featuress = torch.stack(logvar_featuress, dim=1)
        z_featuress = torch.stack(z_featuress, dim=1)
        cropped_objectss = torch.stack(cropped_objectss, dim=1)
        obj_on_as = torch.stack(obj_on_as, dim=1)
        obj_on_bs = torch.stack(obj_on_bs, dim=1)
        z_obj_ons = torch.stack(z_obj_ons, dim=1)
        mu_depths = torch.stack(mu_depths, dim=1)
        logvar_depths = torch.stack(logvar_depths, dim=1)
        z_depths = torch.stack(z_depths, dim=1)
        mu_scales = torch.stack(mu_scales, dim=1)
        logvar_scales = torch.stack(logvar_scales, dim=1)
        z_scales = torch.stack(z_scales, dim=1)

        # decode
        if decode:
            # reshape to [bs * timestep_horizon, ...]
            zs_dec = zs.view(-1, *zs.shape[2:])
            z_featuress_dec = z_featuress.view(-1, *z_featuress.shape[2:])
            z_obj_ons_dec = z_obj_ons.view(-1, *z_obj_ons.shape[2:])
            z_depths_dec = z_depths.view(-1, *z_depths.shape[2:])
            z_scales_dec = z_scales.view(-1, *z_scales.shape[2:])
            decoder_out = self.fg_module.decode_all(zs_dec, z_featuress_dec, z_obj_ons_dec, z_depths_dec, noisy=noisy,
                                                    z_scale=z_scales_dec)
            dec_objectss = decoder_out['dec_objects']
            dec_objects_transs = decoder_out['dec_objects_trans']
            bg_masks = decoder_out['bg_mask']
            alpha_maskss = decoder_out['alpha_masks']
        else:
            dec_objectss = None
            dec_objects_transs = None
            bg_masks = None
            alpha_maskss = None

        if reshape:
            # reshape to [bs * timestep_horizon, ...]
            kp_ps = kp_ps.view(-1, *kp_ps.shape[2:])
            mus = mus.view(-1, *mus.shape[2:])
            logvars = logvars.view(-1, *logvars.shape[2:])
            z_bases = z_bases.view(-1, *z_bases.shape[2:])
            zs = zs.view(-1, *zs.shape[2:])
            mu_offsets = mu_offsets.view(-1, *mu_offsets.shape[2:])
            logvar_offsets = logvar_offsets.view(-1, *logvar_offsets.shape[2:])
            mu_featuress = mu_featuress.view(-1, *mu_featuress.shape[2:])
            logvar_featuress = logvar_featuress.view(-1, *logvar_featuress.shape[2:])
            z_featuress = z_featuress.view(-1, *z_featuress.shape[2:])
            cropped_objectss = cropped_objectss.view(-1, *cropped_objectss.shape[2:])
            obj_on_as = obj_on_as.view(-1, *obj_on_as.shape[2:])
            obj_on_bs = obj_on_bs.view(-1, *obj_on_bs.shape[2:])
            z_obj_ons = z_obj_ons.view(-1, *z_obj_ons.shape[2:])
            mu_depths = mu_depths.view(-1, *mu_depths.shape[2:])
            logvar_depths = logvar_depths.view(-1, *logvar_depths.shape[2:])
            z_depths = z_depths.view(-1, *z_depths.shape[2:])
            mu_scales = mu_scales.view(-1, *mu_scales.shape[2:])
            logvar_scales = logvar_scales.view(-1, *logvar_scales.shape[2:])
            z_scales = z_scales.view(-1, *z_scales.shape[2:])
        else:
            if decode:
                # reshape to [bs, timestep_horizon, ...]
                bg_masks = bg_masks.view(-1, timestep_horizon, *bg_masks.shape[1:])
                dec_objectss = dec_objectss.view(-1, timestep_horizon, *dec_objectss.shape[1:])
                dec_objects_transs = dec_objects_transs.view(-1, timestep_horizon, *dec_objects_transs.shape[1:])
                alpha_maskss = alpha_maskss.view(-1, timestep_horizon, *alpha_maskss.shape[1:])

        output_dict = {'kp_p': kp_ps, 'mu': mus, 'logvar': logvars, 'z_base': z_bases, 'z': zs, 'mu_offset': mu_offsets,
                       'logvar_offset': logvar_offsets, 'mu_features': mu_featuress,
                       'logvar_features': logvar_featuress, 'z_features': z_featuress, 'bg_mask': bg_masks,
                       'cropped_objects_original': cropped_objectss, 'obj_on_a': obj_on_as, 'obj_on_b': obj_on_bs,
                       'obj_on': z_obj_ons, 'dec_objects_original': dec_objectss, 'dec_objects': dec_objects_transs,
                       'mu_depth': mu_depths, 'logvar_depth': logvar_depths, 'z_depth': z_depths, 'mu_scale': mu_scales,
                       'logvar_scale': logvar_scales, 'z_scale': z_scales, 'alpha_masks': alpha_maskss}
        return output_dict

    def encode_all(self, x, deterministic=False, noisy=False, warmup=False, bg_masks_from_fg=False, train_prior=True,
                   cropped_objects_prev=None, scale_prev=None, refinement_iter=True):
        # a wrapper function to convert RGB image to latent particles
        if len(x.shape) == 5 and self.use_tracking:
            # tracking
            # x: [batch_size, T, ch, w, h]
            return self.fg_sequential_opt(x, deterministic=deterministic, noisy=noisy, warmup=warmup,
                                          train_prior=train_prior, decode=False)
        # foreground
        # x: [batch_size, ch, w, h]
        filtering_heuristic = 'random' if (warmup or noisy) else self.filtering_heuristic
        kp_p = self.fg_module.encode_prior(x, x_prior=x, filtering_heuristic=filtering_heuristic)
        kp_init = kp_p if train_prior else (0.0 * kp_p + kp_p.detach())  # 0.0 * kp_p is because of distributed training
        if self.filtering_heuristic == 'none':
            # mlp-mixer
            kp_init = self.fg_module.particle_mixer(x, kp_init)

        fg_dict = self.fg_module.encode_all(x, deterministic=deterministic, noisy=noisy, warmup=warmup, kp_init=kp_init,
                                            cropped_objects_prev=cropped_objects_prev, scale_prev=scale_prev,
                                            refinement_iter=refinement_iter)
        mu = fg_dict['mu']
        logvar = fg_dict['logvar']
        z_base = fg_dict['z_base']
        z = fg_dict['z']
        kp_heatmap = fg_dict['kp_heatmap']
        mu_features = fg_dict['mu_features']
        logvar_features = fg_dict['logvar_features']
        z_features = fg_dict['z_features']
        obj_on_a = fg_dict['obj_on_a']
        obj_on_b = fg_dict['obj_on_b']
        z_obj_on = fg_dict['obj_on']
        mu_depth = fg_dict['mu_depth']
        logvar_depth = fg_dict['logvar_depth']
        z_depth = fg_dict['z_depth']
        cropped_objects = fg_dict['cropped_objects']
        mu_scale = fg_dict['mu_scale']
        logvar_scale = fg_dict['logvar_scale']
        z_scale = fg_dict['z_scale']
        mu_offset = fg_dict['mu_offset']
        logvar_offset = fg_dict['logvar_offset']
        z_offset = fg_dict['z_offset']

        # background
        if bg_masks_from_fg:
            with torch.no_grad():
                fg_dec_out = self.fg_module.decode_all(z, z_features, z_obj_on, z_depth=z_depth, noisy=noisy,
                                                       z_scale=z_scale)
            bg_mask = fg_dec_out['bg_mask']
        else:
            bg_mask = self.get_bg_mask_from_particle_glimpses(z, z_obj_on, mask_size=x.shape[-1])
        bg_dict = self.bg_module.encode_all(x, bg_mask, deterministic)
        mu_bg = bg_dict['mu_bg']
        logvar_bg = bg_dict['logvar_bg']
        z_bg = bg_dict['z_bg']
        z_kp_bg = bg_dict['z_kp']

        encode_dict = {'mu': mu, 'logvar': logvar, 'z': z, 'z_base': z_base, 'kp_heatmap': kp_heatmap,
                       'mu_features': mu_features,
                       'logvar_features': logvar_features, 'z_features': z_features,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'obj_on': z_obj_on,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'cropped_objects': cropped_objects, 'bg_mask': bg_mask,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
                       'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset, 'mu_bg': mu_bg,
                       'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp_bg': z_kp_bg}

        return encode_dict

    def decode_all(self, z, z_features, z_bg, obj_on, z_depth=None, noisy=False, z_scale=None):
        # a wrapper function to convert latent particles to an RGB image
        # foreground
        fg_dict = self.fg_module.decode_all(z, z_features, obj_on, z_depth=z_depth, noisy=noisy, z_scale=z_scale)
        dec_objects = fg_dict['dec_objects']
        dec_objects_trans = fg_dict['dec_objects_trans']
        alpha_masks = fg_dict['alpha_masks']
        bg_mask = fg_dict['bg_mask']
        # background
        bg = self.bg_module.decode_all(z_bg)
        rec = bg_mask * bg + dec_objects_trans
        decoder_out = {'rec': rec, 'dec_objects': dec_objects, 'dec_objects_trans': dec_objects_trans,
                       'bg': bg, 'alpha_masks': alpha_masks}

        return decoder_out

    def forward(self, x, deterministic=False, bg_masks_from_fg=False, x_prior=None, warmup=False, noisy=False,
                train_enc_prior=True):
        if len(x.shape) == 5 and self.use_tracking:
            # tracking
            # x: [batch_size, T, ch, w, h]
            fg_dict = self.fg_sequential_opt(x, deterministic=deterministic, noisy=noisy, warmup=warmup,
                                             train_prior=train_enc_prior, decode=True)
            x = x.view(-1, *x.shape[2:])  # x: [batch_size * T, ch, w, h]
        else:
            # x: [batch_size, ch, w, h]
            fg_dict = self.fg_module(x, deterministic=deterministic, x_prior=x_prior, warmup=warmup, noisy=noisy,
                                     train_prior=train_enc_prior, refinement_iter=True)
        # encoder
        kp_p = fg_dict['kp_p']
        mu = fg_dict['mu']
        logvar = fg_dict['logvar']
        z_base = fg_dict['z_base']
        z = fg_dict['z']
        mu_offset = fg_dict['mu_offset']
        logvar_offset = fg_dict['logvar_offset']
        mu_features = fg_dict['mu_features']
        z_features = fg_dict['z_features']
        logvar_features = fg_dict['logvar_features']
        cropped_objects = fg_dict['cropped_objects_original']
        obj_on_a = fg_dict['obj_on_a']
        obj_on_b = fg_dict['obj_on_b']
        z_obj_on = fg_dict['obj_on']
        mu_depth = fg_dict['mu_depth']
        logvar_depth = fg_dict['logvar_depth']
        z_depth = fg_dict['z_depth']
        mu_scale = fg_dict['mu_scale']
        logvar_scale = fg_dict['logvar_scale']
        z_scale = fg_dict['z_scale']

        # decoder
        bg_mask = fg_dict['bg_mask']
        dec_objects = fg_dict['dec_objects_original']
        dec_objects_trans = fg_dict['dec_objects']
        alpha_masks = fg_dict['alpha_masks']

        if bg_masks_from_fg:
            bg_enc_mask = bg_mask
        else:
            bg_enc_mask = self.get_bg_mask_from_particle_glimpses(z, z_obj_on, mask_size=x.shape[-1])
        bg_dict = self.bg_module(x, bg_enc_mask, deterministic)
        mu_bg = bg_dict['mu_bg']
        logvar_bg = bg_dict['logvar_bg']
        z_bg = bg_dict['z_bg']
        z_kp_bg = bg_dict['z_kp']
        bg = bg_dict['bg_rec']

        # stitch
        rec = bg_mask * bg + dec_objects_trans

        output_dict = {}
        output_dict['kp_p'] = kp_p
        output_dict['rec'] = rec
        output_dict['mu'] = mu
        output_dict['logvar'] = logvar
        output_dict['z'] = z
        output_dict['z_base'] = z_base
        output_dict['z_kp_bg'] = z_kp_bg
        output_dict['mu_offset'] = mu_offset
        output_dict['logvar_offset'] = logvar_offset
        output_dict['mu_features'] = mu_features
        output_dict['logvar_features'] = logvar_features
        output_dict['z_features'] = z_features
        output_dict['bg'] = bg
        output_dict['mu_bg'] = mu_bg
        output_dict['logvar_bg'] = logvar_bg
        output_dict['z_bg'] = z_bg
        # object stuff
        output_dict['cropped_objects_original'] = cropped_objects
        output_dict['obj_on_a'] = obj_on_a
        output_dict['obj_on_b'] = obj_on_b
        output_dict['obj_on'] = z_obj_on
        output_dict['dec_objects_original'] = dec_objects
        output_dict['dec_objects'] = dec_objects_trans
        output_dict['mu_depth'] = mu_depth
        output_dict['logvar_depth'] = logvar_depth
        output_dict['z_depth'] = z_depth
        output_dict['mu_scale'] = mu_scale
        output_dict['logvar_scale'] = logvar_scale
        output_dict['z_scale'] = z_scale
        output_dict['alpha_masks'] = alpha_masks

        return output_dict

    def calc_elbo(self, x, model_output, warmup=False, beta_kl=0.05, beta_rec=1.0, kl_balance=0.001,
                  recon_loss_type="mse", recon_loss_func=None, noisy=False):
        # x: [batch_size, timestep_horizon, ch, h, w]
        # define losses
        kl_loss_func = ChamferLossKL(use_reverse_kl=False)
        if recon_loss_type == "vgg":
            if recon_loss_func is None:
                recon_loss_func = VGGDistance(device=x.device)
        else:
            recon_loss_func = calc_reconstruction_loss

        # unpack output
        mu_p = model_output['kp_p']
        mu = model_output['mu']
        logvar = model_output['logvar']
        z = model_output['z']
        z_base = model_output['z_base']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec']
        mu_features = model_output['mu_features']
        logvar_features = model_output['logvar_features']
        z_features = model_output['z_features']
        mu_bg = model_output['mu_bg']
        logvar_bg = model_output['logvar_bg']
        z_bg = model_output['z_bg']
        mu_scale = model_output['mu_scale']
        logvar_scale = model_output['logvar_scale']
        z_scale = model_output['z_scale']
        mu_depth = model_output['mu_depth']
        logvar_depth = model_output['logvar_depth']
        z_depth = model_output['z_depth']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        cropped_objects_original = model_output['cropped_objects_original']
        obj_on = model_output['obj_on']  # [batch_size, n_kp]
        obj_on_a = model_output['obj_on_a']  # [batch_size, n_kp]
        obj_on_b = model_output['obj_on_b']  # [batch_size, n_kp]
        alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]

        batch_size = x.shape[0]
        if len(x.shape) == 5:
            x = x.view(-1, *x.shape[2:])

        # --- reconstruction error --- #
        if dec_objects_original is not None and warmup:
            if recon_loss_type == "vgg":
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.reshape(-1,
                                                                            *cropped_objects_original.shape[2:])
                if cropped_objects_original.shape[-1] < 32:
                    cropped_objects_original = F.interpolate(cropped_objects_original, size=32, mode='bilinear',
                                                             align_corners=False)
                    dec_objects_rgb = F.interpolate(dec_objects_rgb, size=32, mode='bilinear',
                                                    align_corners=False)
                loss_rec_obj = recon_loss_func(cropped_objects_original, dec_objects_rgb, reduction="mean")

            else:
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.clone().reshape(-1,
                                                                                    *cropped_objects_original.shape[
                                                                                     2:])
                loss_rec_obj = calc_reconstruction_loss(cropped_objects_original, dec_objects_rgb,
                                                        loss_type='mse', reduction='mean')
            loss_rec = loss_rec_obj + (0 * rec_x).mean()  # + (0 * rec_x).mean() for distributed training
            psnr = torch.tensor(0.0, dtype=torch.float, device=x.device)
        else:
            if recon_loss_type == "vgg":
                loss_rec = recon_loss_func(x, rec_x, reduction="mean")
            else:
                loss_rec = calc_reconstruction_loss(x, rec_x, loss_type='mse', reduction='mean')

            with torch.no_grad():
                psnr = -10 * torch.log10(F.mse_loss(rec_x, x))
        # --- end reconstruction error --- #

        # --- define priors --- #
        warmup_logvar = torch.log(torch.tensor(0.1 ** 2))
        logvar_kp = self.logvar_kp.expand_as(mu_p)
        logvar_offset_p = warmup_logvar if (warmup or noisy) else self.logvar_offset_p
        logvar_scale_p = warmup_logvar if (warmup or noisy) else self.logvar_scale_p
        # encourage objects to be 'on' during warmup
        obj_on_a_prior = torch.tensor(0.2, device=obj_on_a.device) if (warmup or noisy) else self.obj_on_a_p
        obj_on_b_prior = torch.tensor(0.1, device=obj_on_b.device) if (warmup or noisy) else self.obj_on_b_p
        # as the scale is sigmoid-activated, we want the mean to be the inverse of the sigmoid of the glimpse size
        mu_scale_prior = self.mu_scale_prior

        # --- end priors --- #

        # kl-divergence and priors
        mu_prior = mu_p
        logvar_prior = logvar_kp
        mu_post = mu
        logvar_post = torch.zeros_like(logvar)
        loss_kl_kp_base = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
                                       logvar_gts=logvar_prior)
        loss_kl_kp_base = loss_kl_kp_base.mean()
        loss_kl_kp_offset = calc_kl(logvar_offset.view(-1, logvar_offset.shape[-1]),
                                    mu_offset.view(-1, mu_offset.shape[-1]), logvar_o=logvar_offset_p,
                                    reduce='none')
        loss_kl_kp_offset = (loss_kl_kp_offset.view(-1, self.n_kp_enc)).sum(-1).mean()
        loss_kl_kp = 0.5 * kl_balance * loss_kl_kp_base + loss_kl_kp_offset

        # depth
        loss_kl_depth = calc_kl(logvar_depth.view(-1, logvar_depth.shape[-1]),
                                mu_depth.view(-1, mu_depth.shape[-1]), reduce='none')
        loss_kl_depth = (loss_kl_depth.view(-1, self.n_kp_enc)).sum(-1).mean()

        # scale
        # assume sigmoid activation on z_scale
        loss_kl_scale = calc_kl(logvar_scale.view(-1, logvar_scale.shape[-1]),
                                mu_scale.view(-1, mu_scale.shape[-1]), mu_o=mu_scale_prior, logvar_o=logvar_scale_p,
                                reduce='none')

        loss_kl_scale = (loss_kl_scale.view(-1, self.n_kp_enc)).sum(-1).mean()

        # obj_on
        loss_kl_obj_on = calc_kl_beta_dist(obj_on_a, obj_on_b,
                                           obj_on_a_prior,
                                           obj_on_b_prior).sum(-1)
        loss_kl_obj_on = loss_kl_obj_on.mean()
        obj_on_l1 = torch.abs(obj_on).sum(-1).mean()

        # features
        loss_kl_feat = calc_kl(logvar_features.view(-1, logvar_features.shape[-1]),
                               mu_features.view(-1, mu_features.shape[-1]), reduce='none')
        loss_kl_feat_obj = loss_kl_feat.view(-1, self.n_kp_enc)
        loss_kl_feat_obj = loss_kl_feat_obj.sum(-1).mean()

        loss_kl_feat_bg = calc_kl(logvar_bg.view(-1, logvar_bg.shape[-1]),
                                  mu_bg.view(-1, mu_bg.shape[-1]), reduce='none')
        loss_kl_feat_bg = loss_kl_feat_bg.mean()
        loss_kl_feat = loss_kl_feat_obj + loss_kl_feat_bg

        loss_kl = loss_kl_kp + loss_kl_scale + loss_kl_obj_on + kl_balance * (loss_kl_feat + loss_kl_depth)

        loss = beta_rec * loss_rec + beta_kl * loss_kl
        if recon_loss_type == "vgg":
            loss = 1e-3 * loss
        loss_dict = {'loss': loss, 'psnr': psnr.detach(), 'kl': loss_kl, 'loss_rec': loss_rec,
                     'obj_on_l1': obj_on_l1, 'loss_kl_kp': loss_kl_kp, 'loss_kl_feat': loss_kl_feat,
                     'loss_kl_obj_on': loss_kl_obj_on, 'loss_kl_scale': loss_kl_scale, 'loss_kl_depth': loss_kl_depth}
        return loss_dict

    def lerp(self, other, betta):
        # weight interpolation for ema - not used in the paper
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = self.parameters()
            other_param = other.parameters()
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)


# video deep latent particles - deep dynamic latent particles (ddlp)
class ObjectDynamicsDLP(nn.Module):
    def __init__(self, cdim=3, enc_channels=(16, 16, 32), prior_channels=(16, 16, 32), image_size=64, n_kp=1,
                 pad_mode='replicate', sigma=1.0, dropout=0.0,
                 patch_size=16, n_kp_enc=20, n_kp_prior=20, learned_feature_dim=16, bg_learned_feature_dim=None,
                 kp_range=(-1, 1), kp_activation="tanh", anchor_s=0.25, predict_delta=True,
                 timestep_horizon=10, enable_enc_attn=False, use_correlation_heatmaps=True,
                 use_resblock=False, scale_std=0.3, offset_std=0.2, obj_on_alpha=0.1, obj_on_beta=0.1, pint_layers=6,
                 pint_heads=8, pint_dim=256, filtering_heuristic='variance'):
        super(ObjectDynamicsDLP, self).__init__()
        """
        cdim: channels of the input image (3...)
        enc_channels: channels for the posterior CNN (takes in the whole image)
        prior_channels: channels for prior CNN (takes in patches)
        n_kp: number of kp to extract from each (!) patch
        n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
        n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        sigma: the prior std of the KP
        dropout: dropout for the CNNs. We don't use it though...
        patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
        kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        bg_learned_feature_dim: the latent visual features dimensions extracted from masked bg. None -> =learned_feature_dim
        kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
        anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
        scale_std: prior std for the scale
        offset_std: prior std for the offset
        obj_on_alpha: prior alpha (Beta distribution) for obj_on
        obj_on_beta: prior beta (Beta distribution) for obj_on
        pint_layers: number of transformer layers in the dynamics module
        pint_heads: number of transformer heads in the dynamics module
        enable_enc_attn: enable attention between patches in the particle encoder
        use_correlation_heatmaps: use correlation heatmaps for tracking
        filtering heuristic: filtering heuristic to filter prior keypoints,['distance', 'variance', 'random', 'none']
        """
        self.image_size = image_size
        self.sigma = sigma
        self.dropout = dropout
        self.kp_range = kp_range
        self.num_patches = int((image_size // patch_size) ** 2)
        self.n_kp = n_kp
        self.n_kp_total = self.n_kp * self.num_patches
        self.n_kp_prior = min(self.n_kp_total, n_kp_prior)
        self.n_kp_enc = n_kp_enc
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.features_dim = int(image_size // (2 ** (len(enc_channels) - 1)))
        self.learned_feature_dim = learned_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.bg_learned_feature_dim = learned_feature_dim if bg_learned_feature_dim is None else bg_learned_feature_dim
        assert self.bg_learned_feature_dim > 0, "bg_learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        # print(f'object patch size: {self.obj_patch_size}')
        self.enable_enc_attn = enable_enc_attn
        self.use_correlation_heatmpas = use_correlation_heatmaps
        self.cdim = cdim
        self.timestep_horizon = timestep_horizon
        self.predict_delta = predict_delta
        self.use_resblock = use_resblock
        self.pint_layers = pint_layers
        self.pint_heads = pint_heads
        self.pint_dim = pint_dim
        assert filtering_heuristic in ['distance', 'variance',
                                       'random', 'none'], f'unknown filtering heuristic: {filtering_heuristic}'
        self.filtering_heuristic = filtering_heuristic

        # priors
        self.register_buffer('logvar_kp', torch.log(torch.tensor(sigma ** 2)))
        self.register_buffer('mu_scale_prior',
                             torch.tensor(np.log(0.75 * self.anchor_s / (1 - 0.75 * self.anchor_s + 1e-5))))
        self.register_buffer('logvar_scale_p', torch.log(torch.tensor(scale_std ** 2)))
        self.register_buffer('logvar_offset_p', torch.log(torch.tensor(offset_std ** 2)))
        self.register_buffer('obj_on_a_p', torch.tensor(obj_on_alpha))
        self.register_buffer('obj_on_b_p', torch.tensor(obj_on_beta))

        # foreground module
        self.fg_module = FgDLP(cdim=cdim, enc_channels=enc_channels, prior_channels=prior_channels,
                               image_size=image_size, n_kp=n_kp, pad_mode=pad_mode,
                               sigma=sigma, dropout=dropout, patch_size=patch_size, n_kp_enc=n_kp_enc,
                               n_kp_prior=n_kp_prior, learned_feature_dim=learned_feature_dim, kp_range=kp_range,
                               kp_activation=kp_activation, anchor_s=anchor_s,
                               use_resblock=self.use_resblock, use_correlation_heatmaps=use_correlation_heatmaps,
                               enable_enc_attn=self.enable_enc_attn, filtering_heuristic=filtering_heuristic)
        # background module
        self.bg_module = BgDLP(cdim=cdim, enc_channels=enc_channels, image_size=image_size, pad_mode=pad_mode,
                               dropout=dropout, learned_feature_dim=self.bg_learned_feature_dim, n_kp_enc=n_kp_enc,
                               use_resblock=self.use_resblock)

        # dynamics module
        max_particles = n_kp_enc + 1  # particle positional bias, +1 for the bg particle
        pint_inner_dim = self.pint_dim
        self.dyn_module = DynamicsDLP(learned_feature_dim, self.bg_learned_feature_dim, hidden_dim=pint_inner_dim,
                                      projection_dim=pint_inner_dim,
                                      n_head=self.pint_heads, n_layer=self.pint_layers, block_size=timestep_horizon,
                                      kp_activation=kp_activation, predict_delta=self.predict_delta, max_delta=1.0,
                                      positional_bias=True, max_particles=max_particles)
        self.init_weights()

    def get_parameters(self, prior=True, encoder=True, decoder=True, dynamics=True):
        parameters = []
        parameters.extend(self.fg_module.get_parameters(prior, encoder, decoder))
        parameters.extend(self.bg_module.get_parameters(prior, encoder, decoder))
        if dynamics:
            parameters.extend(self.dyn_module.parameters())
        return parameters

    def set_require_grad(self, prior_value=True, enc_value=True, dec_value=True):
        self.fg_module.set_require_grad(prior_value, enc_value, dec_value)
        self.bg_module.set_require_grad(prior_value, enc_value, dec_value)

    def init_weights(self):
        self.fg_module.init_weights()
        self.bg_module.init_weights()
        self.fg_module.prior.reset_parameters()

    def info(self):
        log_str = f'prior (patch) kp filtering: {self.n_kp_total} -> prior kp: {self.n_kp_prior}\n'
        log_str += f'prior (patch) kp filtering method: {self.filtering_heuristic}\n'
        log_str += f'prior patch size: {self.patch_size}\n'
        log_str += f'# posterior particles: {self.n_kp_enc}\n'
        log_str += f'particle visual feature dim: {self.learned_feature_dim}\n'
        log_str += f'bg visual feature dim: {self.bg_learned_feature_dim}\n'
        log_str += f'posterior object patch size: {self.obj_patch_size}\n'
        # cnns output sizes
        # prior
        prior_cnn_size = self.fg_module.prior.enc.conv_output_size
        # posterior - objects
        fg_cnn_size = self.fg_module.particle_attribute_enc.cnn.conv_output_size
        # posterior - bg
        bg_cnn_size = self.bg_module.bg_cnn_enc.conv_output_size
        log_str += f'cnn pre-pool out sizes: prior-{prior_cnn_size}, obj enc-{fg_cnn_size}, bg enc-{bg_cnn_size}\n'
        # object decoder
        dec_upsample_l = int(np.log(self.fg_module.object_dec.patch_size[0]) // np.log(2)) - 3
        log_str += f'object decoder # upsampling layers: {dec_upsample_l}\n'
        # tracking
        log_str += f'correlation maps for tracking: {self.fg_module.use_correlation_heatmaps}\n'
        # dynamic module
        log_str += f'pint relative positional bias: {self.dyn_module.particle_transformer.positional_bias}\n'
        pint_size_dict = calc_model_size(self.dyn_module)
        pint_n_params = pint_size_dict['n_params']
        log_str += f'pint trainable parameters: {pint_n_params} ({pint_n_params / (10 ** 6):.4f}M)\n'
        # num parameters and model size
        size_dict = calc_model_size(self)
        size_mb = size_dict['size_mb']
        n_params = size_dict['n_params']
        log_str += f'trainable parameters: {n_params} ({n_params / (10 ** 6):.4f}M)\n'
        log_str += f'estimated size on disk: {size_mb:.3f}MB\n'
        return log_str

    def get_bg_mask_from_particle_glimpses(self, z, z_obj_on, mask_size):
        """
        generates a mask based on particles position and the scale. Masks are squares.
        """
        obj_fmap_masks = create_masks_fast(z.detach(), anchor_s=self.anchor_s, feature_dim=mask_size)
        obj_fmap_masks = obj_fmap_masks.clamp(0, 1) * z_obj_on[:, :, None, None, None].detach()
        bg_mask = 1 - obj_fmap_masks.squeeze(2).sum(1, keepdim=True).clamp(0, 1)
        return bg_mask

    def encode_all(self, x, deterministic=False, x_prior=None, warmup=False, noisy=False, reshape=True,
                   train_prior=False, num_static_frames=4):
        """
        sequential encoding per-timestep, for tracking purposes
        reshape: whether to reshape to [bs * timestep_horizon, ...]
        train_prior: should the prior (proposal) keypoints have gradients
        num_static_frames: the number of frames that are optimized w.r.t the constant prior and not the dynamics
        """
        # a wrapper function to convert RGB image to latent particles
        # x: [bs, T + 1, ...]
        batch_size, timestep_horizon = x.size(0), x.size(1)
        num_static_frames = min(num_static_frames, timestep_horizon)
        if x_prior is None:
            x_prior = x
        # for the first time step, standard encoding
        filtering_heuristic = 'random' if (warmup or noisy) else self.filtering_heuristic
        kp_p = self.fg_module.encode_prior(x[:, :num_static_frames].reshape(-1, *x.shape[2:]),
                                           x_prior=x_prior[:, :num_static_frames].reshape(-1, *x_prior.shape[2:]),
                                           filtering_heuristic=filtering_heuristic)
        kp_p = kp_p.reshape(batch_size, num_static_frames, *kp_p.shape[1:])  # [bs, n_stat_frames, n_kp_p, 2]

        kp_p = kp_p if train_prior else kp_p.detach()
        kp_init = kp_p[:, 0]  # first timestep
        if self.filtering_heuristic == 'none':
            # n_kp_prior -> n_kp_enc
            kp_init = self.particle_mixer(x[:, 0], kp_init)
        fg_dict = self.fg_module.encode_all(x[:, 0], deterministic=deterministic, warmup=warmup, noisy=noisy,
                                            kp_init=kp_init, refinement_iter=True)
        kp_p = kp_p.detach()  # freeze w.r.t to kl-divergence
        mu = fg_dict['mu']
        logvar = fg_dict['logvar']
        z_base = fg_dict['z_base']
        z = fg_dict['z']
        mu_offset = fg_dict['mu_offset']
        logvar_offset = fg_dict['logvar_offset']
        mu_features = fg_dict['mu_features']
        logvar_features = fg_dict['logvar_features']
        z_features = fg_dict['z_features']
        cropped_objects = fg_dict['cropped_objects']
        obj_on_a = fg_dict['obj_on_a']
        obj_on_b = fg_dict['obj_on_b']
        z_obj_on = fg_dict['obj_on']
        mu_depth = fg_dict['mu_depth']
        logvar_depth = fg_dict['logvar_depth']
        z_depth = fg_dict['z_depth']
        mu_scale = fg_dict['mu_scale']
        logvar_scale = fg_dict['logvar_scale']
        z_scale = fg_dict['z_scale']

        # initialize lists to collect all outputs
        mus, logvars, zs, z_bases = [mu], [logvar], [z], [z_base]
        mu_offsets, logvar_offsets = [mu_offset], [logvar_offset]
        mu_featuress, logvar_featuress, z_featuress = [mu_features], [logvar_features], [z_features]
        cropped_objectss = [cropped_objects]
        obj_on_as, obj_on_bs, z_obj_ons = [obj_on_a], [obj_on_b], [z_obj_on]
        mu_depths, logvar_depths, z_depths = [mu_depth], [logvar_depth], [z_depth]
        mu_scales, logvar_scales, z_scales = [mu_scale], [logvar_scale], [z_scale]

        for i in range(1, timestep_horizon):
            # tracking, search for mu_tot in the area of the previous mu
            mu_prev = zs[-1].detach()
            cropped_objects_prev = cropped_objectss[-1].detach()
            cropped_objects_prev = cropped_objects_prev.view(-1, *cropped_objects_prev.shape[2:])
            mu_scale_prev = z_scales[-1].detach()
            fg_dict = self.fg_module.encode_all(x[:, i], deterministic=deterministic, warmup=warmup,
                                                noisy=noisy, kp_init=mu_prev,
                                                cropped_objects_prev=cropped_objects_prev, scale_prev=mu_scale_prev)
            mu = fg_dict['mu']
            logvar = fg_dict['logvar']
            z_base = fg_dict['z_base']
            z = fg_dict['z']
            mu_offset = fg_dict['mu_offset']
            logvar_offset = fg_dict['logvar_offset']
            mu_features = fg_dict['mu_features']
            logvar_features = fg_dict['logvar_features']
            z_features = fg_dict['z_features']
            cropped_objects = fg_dict['cropped_objects']
            obj_on_a = fg_dict['obj_on_a']
            obj_on_b = fg_dict['obj_on_b']
            z_obj_on = fg_dict['obj_on']
            mu_depth = fg_dict['mu_depth']
            logvar_depth = fg_dict['logvar_depth']
            z_depth = fg_dict['z_depth']
            mu_scale = fg_dict['mu_scale']
            logvar_scale = fg_dict['logvar_scale']
            z_scale = fg_dict['z_scale']

            mus.append(mu)
            logvars.append(logvar)
            z_bases.append(z_base)
            zs.append(z)
            mu_offsets.append(mu_offset)
            logvar_offsets.append(logvar_offset)
            mu_featuress.append(mu_features)
            logvar_featuress.append(logvar_features)
            z_featuress.append(z_features)
            cropped_objectss.append(cropped_objects)
            obj_on_as.append(obj_on_a)
            obj_on_bs.append(obj_on_b)
            z_obj_ons.append(z_obj_on)
            mu_depths.append(mu_depth)
            logvar_depths.append(logvar_depth)
            z_depths.append(z_depth)
            mu_scales.append(mu_scale)
            logvar_scales.append(logvar_scale)
            z_scales.append(z_scale)

        # pad the kp proposals (prior) tensor, only care about t=[0, num_static_frames]
        num_pad = len(mus) - kp_p.shape[1]
        if num_pad > 0:
            kp_p_pad = kp_p[:, -1:].detach()
            kp_p = torch.cat([kp_p, kp_p_pad.repeat(1, num_pad, 1, 1)], dim=1)
        kp_ps = kp_p
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)
        z_bases = torch.stack(z_bases, dim=1)
        zs = torch.stack(zs, dim=1)
        mu_offsets = torch.stack(mu_offsets, dim=1)
        logvar_offsets = torch.stack(logvar_offsets, dim=1)
        mu_featuress = torch.stack(mu_featuress, dim=1)
        logvar_featuress = torch.stack(logvar_featuress, dim=1)
        z_featuress = torch.stack(z_featuress, dim=1)
        cropped_objectss = torch.stack(cropped_objectss, dim=1)
        obj_on_as = torch.stack(obj_on_as, dim=1)
        obj_on_bs = torch.stack(obj_on_bs, dim=1)
        z_obj_ons = torch.stack(z_obj_ons, dim=1)
        mu_depths = torch.stack(mu_depths, dim=1)
        logvar_depths = torch.stack(logvar_depths, dim=1)
        z_depths = torch.stack(z_depths, dim=1)
        mu_scales = torch.stack(mu_scales, dim=1)
        logvar_scales = torch.stack(logvar_scales, dim=1)
        z_scales = torch.stack(z_scales, dim=1)

        if reshape:
            # reshape to [bs * timestep_horizon, ...]
            kp_ps = kp_ps.view(-1, *kp_ps.shape[2:])
            mus = mus.view(-1, *mus.shape[2:])
            logvars = logvars.view(-1, *logvars.shape[2:])
            z_bases = z_bases.view(-1, *z_bases.shape[2:])
            zs = zs.view(-1, *zs.shape[2:])
            mu_offsets = mu_offsets.view(-1, *mu_offsets.shape[2:])
            logvar_offsets = logvar_offsets.view(-1, *logvar_offsets.shape[2:])
            mu_featuress = mu_featuress.view(-1, *mu_featuress.shape[2:])
            logvar_featuress = logvar_featuress.view(-1, *logvar_featuress.shape[2:])
            z_featuress = z_featuress.view(-1, *z_featuress.shape[2:])
            cropped_objectss = cropped_objectss.view(-1, *cropped_objectss.shape[2:])
            obj_on_as = obj_on_as.view(-1, *obj_on_as.shape[2:])
            obj_on_bs = obj_on_bs.view(-1, *obj_on_bs.shape[2:])
            z_obj_ons = z_obj_ons.view(-1, *z_obj_ons.shape[2:])
            mu_depths = mu_depths.view(-1, *mu_depths.shape[2:])
            logvar_depths = logvar_depths.view(-1, *logvar_depths.shape[2:])
            z_depths = z_depths.view(-1, *z_depths.shape[2:])
            mu_scales = mu_scales.view(-1, *mu_scales.shape[2:])
            logvar_scales = logvar_scales.view(-1, *logvar_scales.shape[2:])
            z_scales = z_scales.view(-1, *z_scales.shape[2:])

        output_dict = {'kp_p': kp_ps, 'mu': mus, 'logvar': logvars, 'z_base': z_bases, 'z': zs, 'mu_offset': mu_offsets,
                       'logvar_offset': logvar_offsets, 'mu_features': mu_featuress,
                       'logvar_features': logvar_featuress, 'z_features': z_featuress,
                       'cropped_objects_original': cropped_objectss, 'obj_on_a': obj_on_as, 'obj_on_b': obj_on_bs,
                       'obj_on': z_obj_ons,
                       'mu_depth': mu_depths, 'logvar_depth': logvar_depths, 'z_depth': z_depths, 'mu_scale': mu_scales,
                       'logvar_scale': logvar_scales, 'z_scale': z_scales}
        return output_dict

    def decode_all(self, z, z_features, z_bg, obj_on, z_depth=None, noisy=False, z_scale=None):
        # a wrapper function to convert latent particles to an RGB image
        # foreground
        fg_dict = self.fg_module.decode_all(z, z_features, obj_on, z_depth=z_depth, noisy=noisy, z_scale=z_scale)
        dec_objects = fg_dict['dec_objects']
        dec_objects_trans = fg_dict['dec_objects_trans']
        alpha_masks = fg_dict['alpha_masks']
        bg_mask = fg_dict['bg_mask']
        # background
        bg = self.bg_module.decode_all(z_bg)
        rec = bg_mask * bg + dec_objects_trans
        decoder_out = {'rec': rec, 'dec_objects': dec_objects, 'dec_objects_trans': dec_objects_trans,
                       'bg': bg, 'alpha_masks': alpha_masks}

        return decoder_out

    def sample(self, x, num_steps=10, deterministic=True, bg_masks_from_fg=False, cond_steps=None, return_z=False):
        """
        (Conditioanl) Sampling from DDLP: x is the conditional frames, encoded to latent particles
         which are unrolled to the future with PINT, and the predicted particles are then decoded to a sequence
         of RGB images.
        """
        # x: [bs, T, ...]
        # encode-decode
        batch_size, timestep_horizon_all = x.size(0), x.size(1)
        timestep_horizon = self.timestep_horizon if cond_steps is None else cond_steps

        # sequential
        fg_dict = self.fg_sequential_opt(x[:, :timestep_horizon], deterministic=deterministic,
                                         x_prior=x[:, :timestep_horizon], reshape=True)

        x_in = x[:, :timestep_horizon].reshape(-1, *x.shape[2:])  # [bs * T, ...]
        # encoder
        z = fg_dict['z']
        z_features = fg_dict['z_features']
        z_obj_on = fg_dict['obj_on']
        z_depth = fg_dict['z_depth']
        z_scale = fg_dict['z_scale']

        # decoder
        bg_mask = fg_dict['bg_mask']
        dec_objects_trans = fg_dict['dec_objects']

        if bg_masks_from_fg:
            bg_enc_mask = bg_mask
        else:
            bg_enc_mask = self.get_bg_mask_from_particle_glimpses(z, z_obj_on, mask_size=x_in.shape[-1])
        bg_dict = self.bg_module(x_in, bg_enc_mask, deterministic)
        z_bg = bg_dict['z_bg']
        bg = bg_dict['bg_rec']

        # stitch
        rec = bg_mask * bg + dec_objects_trans
        rec = rec.view(batch_size, -1, *rec.shape[1:])

        z_v = z.view(batch_size, timestep_horizon, *z.shape[1:])
        z_scale_v = z_scale.view(batch_size, timestep_horizon, *z_scale.shape[1:])
        z_obj_on_v = z_obj_on.view(batch_size, timestep_horizon, *z_obj_on.shape[1:])
        z_depth_v = z_depth.view(batch_size, timestep_horizon, *z_depth.shape[1:])
        z_features_v = z_features.view(batch_size, timestep_horizon, *z_features.shape[1:])
        z_bg_features_v = z_bg.view(batch_size, timestep_horizon, *z_bg.shape[1:])

        # dynamics
        dyn_out = self.dyn_module.sample(z_v, z_scale_v, z_obj_on_v, z_depth_v, z_features_v, z_bg_features_v,
                                         steps=num_steps, deterministic=deterministic)
        z_dyn, z_scale_dyn, z_obj_on_dyn, z_depth_dyn, z_features_dyn, z_bg_features_dyn = dyn_out
        if return_z:
            z_ids = 1 + torch.arange(z_dyn.shape[2], device=z_dyn.device)  # num_particles, ids start from 1
            z_ids = z_ids[None, None, :].repeat(z_dyn.shape[0], z_dyn.shape[1], 1)  # [bs, T, n_particles]
            z_out = {'z_pos': z_dyn.detach(), 'z_scale': z_scale_dyn.detach(), 'z_obj_on': z_obj_on_dyn.detach(),
                     'z_depth': z_depth_dyn.detach(), 'z_features': z_features_dyn.detach(),
                     'z_bg_features': z_bg_features_dyn.detach(), 'z_ids': z_ids}
        # decode
        z_dyn = z_dyn[:, -num_steps:].reshape(-1, *z_dyn.shape[2:])
        z_features_dyn = z_features_dyn[:, -num_steps:].reshape(-1, *z_features_dyn.shape[2:])
        z_bg_features_dyn = z_bg_features_dyn[:, -num_steps:].reshape(-1, *z_bg_features_dyn.shape[2:])
        z_obj_on_dyn = z_obj_on_dyn[:, -num_steps:].reshape(-1, *z_obj_on_dyn.shape[2:])
        z_depth_dyn = z_depth_dyn[:, -num_steps:].reshape(-1, *z_depth_dyn.shape[2:])
        z_scale_dyn = z_scale_dyn[:, -num_steps:].reshape(-1, *z_scale_dyn.shape[2:])
        dec_out = self.decode_all(z_dyn, z_features_dyn, z_bg_features_dyn, z_obj_on_dyn,
                                  z_depth=z_depth_dyn, z_scale=z_scale_dyn)
        rec_dyn = dec_out['rec']
        rec_dyn = rec_dyn.reshape(batch_size, -1, *rec_dyn.shape[1:])
        rec = torch.cat([rec, rec_dyn], dim=1)
        assert timestep_horizon_all == rec.shape[1], "prediction and gt frames shape don't match"
        if return_z:
            return rec, z_out
        return rec

    def fg_sequential_latent(self, x, deterministic=False, x_prior=None, warmup=False, noisy=False, reshape=True,
                             train_prior=False, num_static_frames=4):
        """
        NOT USED- REMOVE IN FINAL VERSION
        sequential encoding (+ object decoding) per-timestep, for tracking purposes
        reshape: whether to reshape to [bs * timestep_horizon, ...]
        train_prior: should the prior (proposal) keypoints have gradients
        num_static_frames: the number of frames that are optimized w.r.t the constant prior and not the dynamics
                            (burn-in frames)
        here, we are first getting all the latents and then match particles based on latent similarity
        """
        # x: [bs, T + 1, ...]
        batch_size, timestep_horizon = x.size(0), x.size(1)
        x_in = x.reshape(-1, *x.shape[2:])
        num_static_frames = min(num_static_frames, timestep_horizon)
        if x_prior is None:
            x_prior = x_in
        else:
            x_prior = x_prior.reshape(-1, *x_prior.shape[2:])
        filtering_heuristic = self.filtering_heuristic
        kp_p = self.fg_module.encode_prior(x_in, x_prior=x_prior,
                                           filtering_heuristic=filtering_heuristic)  # [bs * (T + 1), n_kp_p, 2]
        kp_p = kp_p if train_prior else (0.0 * kp_p + kp_p.detach())  # 0.0 * kp_p for distributed training
        kp_init = kp_p
        if self.filtering_heuristic == 'none':
            # n_kp_prior -> n_kp_enc
            kp_init = self.particle_mixer(x_in, kp_init)
        # refinement_iter = not (warmup or noisy)
        fg_dict = self.fg_module.encode_all(x_in, deterministic=deterministic, warmup=warmup, noisy=noisy,
                                            kp_init=kp_init, refinement_iter=True)
        kp_p = kp_p.detach()  # freeze w.r.t to kl-divergence
        mu = fg_dict['mu'].view(batch_size, timestep_horizon, *fg_dict['mu'].shape[1:])
        logvar = fg_dict['logvar'].view(batch_size, timestep_horizon, *fg_dict['logvar'].shape[1:])
        z_base = fg_dict['z_base'].view(batch_size, timestep_horizon, *fg_dict['z_base'].shape[1:])
        z = fg_dict['z'].view(batch_size, timestep_horizon, *fg_dict['z'].shape[1:])
        mu_offset = fg_dict['mu_offset'].view(batch_size, timestep_horizon, *fg_dict['mu_offset'].shape[1:])
        logvar_offset = fg_dict['logvar_offset'].view(batch_size, timestep_horizon, *fg_dict['logvar_offset'].shape[1:])
        mu_features = fg_dict['mu_features'].view(batch_size, timestep_horizon, *fg_dict['mu_features'].shape[1:])
        logvar_features = fg_dict['logvar_features'].view(batch_size, timestep_horizon,
                                                          *fg_dict['logvar_features'].shape[1:])
        z_features = fg_dict['z_features'].view(batch_size, timestep_horizon, *fg_dict['z_features'].shape[1:])
        cropped_objects = fg_dict['cropped_objects'].view(batch_size, timestep_horizon,
                                                          *fg_dict['cropped_objects'].shape[1:])
        obj_on_a = fg_dict['obj_on_a'].view(batch_size, timestep_horizon, *fg_dict['obj_on_a'].shape[1:])
        obj_on_b = fg_dict['obj_on_b'].view(batch_size, timestep_horizon, *fg_dict['obj_on_b'].shape[1:])
        z_obj_on = fg_dict['obj_on'].view(batch_size, timestep_horizon, *fg_dict['obj_on'].shape[1:])
        mu_depth = fg_dict['mu_depth'].view(batch_size, timestep_horizon, *fg_dict['mu_depth'].shape[1:])
        logvar_depth = fg_dict['logvar_depth'].view(batch_size, timestep_horizon, *fg_dict['logvar_depth'].shape[1:])
        z_depth = fg_dict['z_depth'].view(batch_size, timestep_horizon, *fg_dict['z_depth'].shape[1:])
        mu_scale = fg_dict['mu_scale'].view(batch_size, timestep_horizon, *fg_dict['mu_scale'].shape[1:])
        logvar_scale = fg_dict['logvar_scale'].view(batch_size, timestep_horizon, *fg_dict['logvar_scale'].shape[1:])
        z_scale = fg_dict['z_scale'].view(batch_size, timestep_horizon, *fg_dict['z_scale'].shape[1:])

        # initialize lists to collect all outputs
        mus, logvars, zs, z_bases = [mu[:, 0]], [logvar[:, 0]], [z[:, 0]], [z_base[:, 0]]
        mu_offsets, logvar_offsets = [mu_offset[:, 0]], [logvar_offset[:, 0]]
        mu_featuress, logvar_featuress, z_featuress = [mu_features[:, 0]], [logvar_features[:, 0]], [z_features[:, 0]]
        cropped_objectss = [cropped_objects[:, 0]]
        obj_on_as, obj_on_bs, z_obj_ons = [obj_on_a[:, 0]], [obj_on_b[:, 0]], [z_obj_on[:, 0]]
        mu_depths, logvar_depths, z_depths = [mu_depth[:, 0]], [logvar_depth[:, 0]], [z_depth[:, 0]]
        mu_scales, logvar_scales, z_scales = [mu_scale[:, 0]], [logvar_scale[:, 0]], [z_scale[:, 0]]

        for i in range(1, timestep_horizon):
            # tracking, iterative latent matching
            with torch.no_grad():
                # previous particle features
                prev_mu = z_bases[-1] + mu_offsets[-1]
                candid_mu = z_base[:, i] + mu_offset[:, i]
                # previous_features = torch.cat(
                #     [prev_mu, mu_scales[-1], mu_depths[-1], z_obj_ons[-1].unsqueeze(-1), mu_featuress[-1]], dim=-1)
                previous_features = torch.cat(
                    [prev_mu, torch.sigmoid(mu_scales[-1])], dim=-1)
                # [batch_size, n_kp_1, features_dim]
                # candidate_features = torch.cat(
                #     [candid_mu, mu_scale[:, i], mu_depth[:, i], z_obj_on[:, i].unsqueeze(-1), mu_features[:, i]],
                #     dim=-1)
                candidate_features = torch.cat(
                    [candid_mu, torch.sigmoid(mu_scale[:, i])], dim=-1)
                # [batch_size, n_kp_2, features_dim]
                # (n_kp_1 = n_kp_2, but for clarity)
                # calculate cost
                # TODO: weighted chamfer-kl cost? with kl_balance for features
                cost = (previous_features.unsqueeze(2) - candidate_features.unsqueeze(1)) ** 2
                # [batch_size, n_kp_1, n_kp_2, features_dim]
                # aggregate cost
                cost = cost.sum(-1)  # [batch_size, n_kp_1, n_kp_2]
                # get min idx
                min_idx = cost.min(dim=-1)[1]  # [batch_size, n_kp_1]

            batch_idx = torch.arange(batch_size, device=x.device)[:, None]

            mus.append(mu[batch_idx, i, min_idx])
            logvars.append(logvar[batch_idx, i, min_idx])
            z_bases.append(z_base[batch_idx, i, min_idx])
            zs.append(z[batch_idx, i, min_idx])
            mu_offsets.append(mu_offset[batch_idx, i, min_idx])
            logvar_offsets.append(logvar_offset[batch_idx, i, min_idx])
            mu_featuress.append(mu_features[batch_idx, i, min_idx])
            logvar_featuress.append(logvar_features[batch_idx, i, min_idx])
            z_featuress.append(z_features[batch_idx, i, min_idx])
            cropped_objectss.append(cropped_objects[batch_idx, i, min_idx])
            obj_on_as.append(obj_on_a[batch_idx, i, min_idx])
            obj_on_bs.append(obj_on_b[batch_idx, i, min_idx])
            z_obj_ons.append(z_obj_on[batch_idx, i, min_idx])
            mu_depths.append(mu_depth[batch_idx, i, min_idx])
            logvar_depths.append(logvar_depth[batch_idx, i, min_idx])
            z_depths.append(z_depth[batch_idx, i, min_idx])
            mu_scales.append(mu_scale[batch_idx, i, min_idx])
            logvar_scales.append(logvar_scale[batch_idx, i, min_idx])
            z_scales.append(z_scale[batch_idx, i, min_idx])

        # pad the kp proposals (prior) tensor, only care about t=[0, num_static_frames]
        # num_pad = len(mus) - kp_p.shape[1]
        # if num_pad > 0:
        #     kp_p_pad = kp_p[:, -1:].detach()
        #     kp_p = torch.cat([kp_p, kp_p_pad.repeat(1, num_pad, 1, 1)], dim=1)
        kp_ps = kp_p
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)
        z_bases = torch.stack(z_bases, dim=1)
        zs = torch.stack(zs, dim=1)
        mu_offsets = torch.stack(mu_offsets, dim=1)
        logvar_offsets = torch.stack(logvar_offsets, dim=1)
        mu_featuress = torch.stack(mu_featuress, dim=1)
        logvar_featuress = torch.stack(logvar_featuress, dim=1)
        z_featuress = torch.stack(z_featuress, dim=1)
        cropped_objectss = torch.stack(cropped_objectss, dim=1)
        obj_on_as = torch.stack(obj_on_as, dim=1)
        obj_on_bs = torch.stack(obj_on_bs, dim=1)
        z_obj_ons = torch.stack(z_obj_ons, dim=1)
        mu_depths = torch.stack(mu_depths, dim=1)
        logvar_depths = torch.stack(logvar_depths, dim=1)
        z_depths = torch.stack(z_depths, dim=1)
        mu_scales = torch.stack(mu_scales, dim=1)
        logvar_scales = torch.stack(logvar_scales, dim=1)
        z_scales = torch.stack(z_scales, dim=1)

        # decode
        # reshape to [bs * timestep_horizon, ...]
        zs_dec = zs.view(-1, *zs.shape[2:])
        z_featuress_dec = z_featuress.view(-1, *z_featuress.shape[2:])
        z_obj_ons_dec = z_obj_ons.view(-1, *z_obj_ons.shape[2:])
        z_depths_dec = z_depths.view(-1, *z_depths.shape[2:])
        z_scales_dec = z_scales.view(-1, *z_scales.shape[2:])
        decoder_out = self.fg_module.decode_all(zs_dec, z_featuress_dec, z_obj_ons_dec, z_depths_dec, noisy=noisy,
                                                z_scale=z_scales_dec)
        dec_objectss = decoder_out['dec_objects']
        dec_objects_transs = decoder_out['dec_objects_trans']
        bg_masks = decoder_out['bg_mask']
        alpha_maskss = decoder_out['alpha_masks']

        if reshape:
            # reshape to [bs * timestep_horizon, ...]
            # kp_ps = kp_ps.view(-1, *kp_ps.shape[2:])
            mus = mus.view(-1, *mus.shape[2:])
            logvars = logvars.view(-1, *logvars.shape[2:])
            z_bases = z_bases.view(-1, *z_bases.shape[2:])
            zs = zs.view(-1, *zs.shape[2:])
            mu_offsets = mu_offsets.view(-1, *mu_offsets.shape[2:])
            logvar_offsets = logvar_offsets.view(-1, *logvar_offsets.shape[2:])
            mu_featuress = mu_featuress.view(-1, *mu_featuress.shape[2:])
            logvar_featuress = logvar_featuress.view(-1, *logvar_featuress.shape[2:])
            z_featuress = z_featuress.view(-1, *z_featuress.shape[2:])
            cropped_objectss = cropped_objectss.view(-1, *cropped_objectss.shape[2:])
            obj_on_as = obj_on_as.view(-1, *obj_on_as.shape[2:])
            obj_on_bs = obj_on_bs.view(-1, *obj_on_bs.shape[2:])
            z_obj_ons = z_obj_ons.view(-1, *z_obj_ons.shape[2:])
            mu_depths = mu_depths.view(-1, *mu_depths.shape[2:])
            logvar_depths = logvar_depths.view(-1, *logvar_depths.shape[2:])
            z_depths = z_depths.view(-1, *z_depths.shape[2:])
            mu_scales = mu_scales.view(-1, *mu_scales.shape[2:])
            logvar_scales = logvar_scales.view(-1, *logvar_scales.shape[2:])
            z_scales = z_scales.view(-1, *z_scales.shape[2:])
        else:
            # reshape to [bs, timestep_horizon, ...]
            bg_masks = bg_masks.view(-1, timestep_horizon, *bg_masks.shape[1:])
            dec_objectss = dec_objectss.view(-1, timestep_horizon, *dec_objectss.shape[1:])
            dec_objects_transs = dec_objects_transs.view(-1, timestep_horizon, *dec_objects_transs.shape[1:])
            alpha_maskss = alpha_maskss.view(-1, timestep_horizon, *alpha_maskss.shape[1:])

        output_dict = {'kp_p': kp_ps, 'mu': mus, 'logvar': logvars, 'z_base': z_bases, 'z': zs, 'mu_offset': mu_offsets,
                       'logvar_offset': logvar_offsets, 'mu_features': mu_featuress,
                       'logvar_features': logvar_featuress, 'z_features': z_featuress, 'bg_mask': bg_masks,
                       'cropped_objects_original': cropped_objectss, 'obj_on_a': obj_on_as, 'obj_on_b': obj_on_bs,
                       'obj_on': z_obj_ons, 'dec_objects_original': dec_objectss, 'dec_objects': dec_objects_transs,
                       'mu_depth': mu_depths, 'logvar_depth': logvar_depths, 'z_depth': z_depths, 'mu_scale': mu_scales,
                       'logvar_scale': logvar_scales, 'z_scale': z_scales, 'alpha_masks': alpha_maskss}
        return output_dict

    def fg_sequential_opt(self, x, deterministic=False, x_prior=None, warmup=False, noisy=False, reshape=True,
                          train_prior=False, num_static_frames=4):
        """
        sequential encoding (+ object decoding) per-timestep, for tracking purposes
        reshape: whether to reshape to [bs * timestep_horizon, ...]
        train_prior: should the prior (proposal) keypoints have gradients
        num_static_frames: the number of frames that are optimized w.r.t the constant prior and not the dynamics
                            (burn-in frames)
        """
        # x: [bs, T + 1, ...]
        batch_size, timestep_horizon = x.size(0), x.size(1)
        num_static_frames = min(num_static_frames, timestep_horizon)
        if x_prior is None:
            x_prior = x
        # for the first time step, standard encoding
        filtering_heuristic = self.filtering_heuristic
        kp_p = self.fg_module.encode_prior(x[:, :num_static_frames].reshape(-1, *x.shape[2:]),
                                           x_prior=x_prior[:, :num_static_frames].reshape(-1, *x_prior.shape[2:]),
                                           filtering_heuristic=filtering_heuristic)
        kp_p = kp_p.reshape(batch_size, num_static_frames, *kp_p.shape[1:])  # [bs, n_stat_frames, n_kp_p, 2]
        kp_p = kp_p if train_prior else (0.0 * kp_p + kp_p.detach())  # 0.0 * kp_p for distributed training
        kp_init = kp_p[:, 0]  # first timestep
        if self.filtering_heuristic == 'none':
            # n_kp_prior -> n_kp_enc
            kp_init = self.particle_mixer(x[:, 0], kp_init)
        fg_dict = self.fg_module.encode_all(x[:, 0], deterministic=deterministic, warmup=warmup, noisy=noisy,
                                            kp_init=kp_init, refinement_iter=True)
        kp_p = kp_p.detach()  # freeze w.r.t to kl-divergence
        mu = fg_dict['mu']
        logvar = fg_dict['logvar']
        z_base = fg_dict['z_base']
        z = fg_dict['z']
        mu_offset = fg_dict['mu_offset']
        logvar_offset = fg_dict['logvar_offset']
        mu_features = fg_dict['mu_features']
        logvar_features = fg_dict['logvar_features']
        z_features = fg_dict['z_features']
        cropped_objects = fg_dict['cropped_objects']
        obj_on_a = fg_dict['obj_on_a']
        obj_on_b = fg_dict['obj_on_b']
        z_obj_on = fg_dict['obj_on']
        mu_depth = fg_dict['mu_depth']
        logvar_depth = fg_dict['logvar_depth']
        z_depth = fg_dict['z_depth']
        mu_scale = fg_dict['mu_scale']
        logvar_scale = fg_dict['logvar_scale']
        z_scale = fg_dict['z_scale']

        # initialize lists to collect all outputs
        mus, logvars, zs, z_bases = [mu], [logvar], [z], [z_base]
        mu_offsets, logvar_offsets = [mu_offset], [logvar_offset]
        mu_featuress, logvar_featuress, z_featuress = [mu_features], [logvar_features], [z_features]
        cropped_objectss = [cropped_objects]
        obj_on_as, obj_on_bs, z_obj_ons = [obj_on_a], [obj_on_b], [z_obj_on]
        mu_depths, logvar_depths, z_depths = [mu_depth], [logvar_depth], [z_depth]
        mu_scales, logvar_scales, z_scales = [mu_scale], [logvar_scale], [z_scale]

        for i in range(1, timestep_horizon):
            # tracking, search for mu_tot in the area of the previous mu
            mu_prev = zs[-1].detach()
            cropped_objects_prev = cropped_objectss[-1].detach()
            cropped_objects_prev = cropped_objects_prev.view(-1, *cropped_objects_prev.shape[2:])
            mu_scale_prev = z_scales[-1].detach()
            fg_dict = self.fg_module.encode_all(x[:, i], deterministic=deterministic, warmup=warmup,
                                                noisy=noisy, kp_init=mu_prev,
                                                cropped_objects_prev=cropped_objects_prev, scale_prev=mu_scale_prev,
                                                refinement_iter=False)
            mu = fg_dict['mu']
            logvar = fg_dict['logvar']
            z_base = fg_dict['z_base']
            z = fg_dict['z']
            mu_offset = fg_dict['mu_offset']
            logvar_offset = fg_dict['logvar_offset']
            mu_features = fg_dict['mu_features']
            logvar_features = fg_dict['logvar_features']
            z_features = fg_dict['z_features']
            cropped_objects = fg_dict['cropped_objects']
            obj_on_a = fg_dict['obj_on_a']
            obj_on_b = fg_dict['obj_on_b']
            z_obj_on = fg_dict['obj_on']
            mu_depth = fg_dict['mu_depth']
            logvar_depth = fg_dict['logvar_depth']
            z_depth = fg_dict['z_depth']
            mu_scale = fg_dict['mu_scale']
            logvar_scale = fg_dict['logvar_scale']
            z_scale = fg_dict['z_scale']

            mus.append(mu)
            logvars.append(logvar)
            z_bases.append(z_base)
            zs.append(z)
            mu_offsets.append(mu_offset)
            logvar_offsets.append(logvar_offset)
            mu_featuress.append(mu_features)
            logvar_featuress.append(logvar_features)
            z_featuress.append(z_features)
            cropped_objectss.append(cropped_objects)
            obj_on_as.append(obj_on_a)
            obj_on_bs.append(obj_on_b)
            z_obj_ons.append(z_obj_on)
            mu_depths.append(mu_depth)
            logvar_depths.append(logvar_depth)
            z_depths.append(z_depth)
            mu_scales.append(mu_scale)
            logvar_scales.append(logvar_scale)
            z_scales.append(z_scale)

        # pad the kp proposals (prior) tensor, only care about t=[0, num_static_frames]
        num_pad = len(mus) - kp_p.shape[1]
        if num_pad > 0:
            kp_p_pad = kp_p[:, -1:].detach()
            kp_p = torch.cat([kp_p, kp_p_pad.repeat(1, num_pad, 1, 1)], dim=1)
        kp_ps = kp_p
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)
        z_bases = torch.stack(z_bases, dim=1)
        zs = torch.stack(zs, dim=1)
        mu_offsets = torch.stack(mu_offsets, dim=1)
        logvar_offsets = torch.stack(logvar_offsets, dim=1)
        mu_featuress = torch.stack(mu_featuress, dim=1)
        logvar_featuress = torch.stack(logvar_featuress, dim=1)
        z_featuress = torch.stack(z_featuress, dim=1)
        cropped_objectss = torch.stack(cropped_objectss, dim=1)
        obj_on_as = torch.stack(obj_on_as, dim=1)
        obj_on_bs = torch.stack(obj_on_bs, dim=1)
        z_obj_ons = torch.stack(z_obj_ons, dim=1)
        mu_depths = torch.stack(mu_depths, dim=1)
        logvar_depths = torch.stack(logvar_depths, dim=1)
        z_depths = torch.stack(z_depths, dim=1)
        mu_scales = torch.stack(mu_scales, dim=1)
        logvar_scales = torch.stack(logvar_scales, dim=1)
        z_scales = torch.stack(z_scales, dim=1)

        # decode
        # reshape to [bs * timestep_horizon, ...]
        zs_dec = zs.view(-1, *zs.shape[2:])
        z_featuress_dec = z_featuress.view(-1, *z_featuress.shape[2:])
        z_obj_ons_dec = z_obj_ons.view(-1, *z_obj_ons.shape[2:])
        z_depths_dec = z_depths.view(-1, *z_depths.shape[2:])
        z_scales_dec = z_scales.view(-1, *z_scales.shape[2:])
        decoder_out = self.fg_module.decode_all(zs_dec, z_featuress_dec, z_obj_ons_dec, z_depths_dec, noisy=noisy,
                                                z_scale=z_scales_dec)
        dec_objectss = decoder_out['dec_objects']
        dec_objects_transs = decoder_out['dec_objects_trans']
        bg_masks = decoder_out['bg_mask']
        alpha_maskss = decoder_out['alpha_masks']

        if reshape:
            # reshape to [bs * timestep_horizon, ...]
            kp_ps = kp_ps.view(-1, *kp_ps.shape[2:])
            mus = mus.view(-1, *mus.shape[2:])
            logvars = logvars.view(-1, *logvars.shape[2:])
            z_bases = z_bases.view(-1, *z_bases.shape[2:])
            zs = zs.view(-1, *zs.shape[2:])
            mu_offsets = mu_offsets.view(-1, *mu_offsets.shape[2:])
            logvar_offsets = logvar_offsets.view(-1, *logvar_offsets.shape[2:])
            mu_featuress = mu_featuress.view(-1, *mu_featuress.shape[2:])
            logvar_featuress = logvar_featuress.view(-1, *logvar_featuress.shape[2:])
            z_featuress = z_featuress.view(-1, *z_featuress.shape[2:])
            cropped_objectss = cropped_objectss.view(-1, *cropped_objectss.shape[2:])
            obj_on_as = obj_on_as.view(-1, *obj_on_as.shape[2:])
            obj_on_bs = obj_on_bs.view(-1, *obj_on_bs.shape[2:])
            z_obj_ons = z_obj_ons.view(-1, *z_obj_ons.shape[2:])
            mu_depths = mu_depths.view(-1, *mu_depths.shape[2:])
            logvar_depths = logvar_depths.view(-1, *logvar_depths.shape[2:])
            z_depths = z_depths.view(-1, *z_depths.shape[2:])
            mu_scales = mu_scales.view(-1, *mu_scales.shape[2:])
            logvar_scales = logvar_scales.view(-1, *logvar_scales.shape[2:])
            z_scales = z_scales.view(-1, *z_scales.shape[2:])
        else:
            # reshape to [bs, timestep_horizon, ...]
            bg_masks = bg_masks.view(-1, timestep_horizon, *bg_masks.shape[1:])
            dec_objectss = dec_objectss.view(-1, timestep_horizon, *dec_objectss.shape[1:])
            dec_objects_transs = dec_objects_transs.view(-1, timestep_horizon, *dec_objects_transs.shape[1:])
            alpha_maskss = alpha_maskss.view(-1, timestep_horizon, *alpha_maskss.shape[1:])

        output_dict = {'kp_p': kp_ps, 'mu': mus, 'logvar': logvars, 'z_base': z_bases, 'z': zs, 'mu_offset': mu_offsets,
                       'logvar_offset': logvar_offsets, 'mu_features': mu_featuress,
                       'logvar_features': logvar_featuress, 'z_features': z_featuress, 'bg_mask': bg_masks,
                       'cropped_objects_original': cropped_objectss, 'obj_on_a': obj_on_as, 'obj_on_b': obj_on_bs,
                       'obj_on': z_obj_ons, 'dec_objects_original': dec_objectss, 'dec_objects': dec_objects_transs,
                       'mu_depth': mu_depths, 'logvar_depth': logvar_depths, 'z_depth': z_depths, 'mu_scale': mu_scales,
                       'logvar_scale': logvar_scales, 'z_scale': z_scales, 'alpha_masks': alpha_maskss}
        return output_dict

    def forward(self, x, deterministic=False, bg_masks_from_fg=False, x_prior=None, warmup=False, noisy=False,
                forward_dyn=True, train_enc_prior=True, num_static_frames=4):
        # x: [bs, T + 1, ...]
        batch_size, timestep_horizon = x.size(0), x.size(1)
        # timestep_horizon = timestep_horizon - 1

        x_in = x.view(-1, *x.shape[2:])  # [bs * T, ...]
        if forward_dyn:
            # tracking
            fg_dict = self.fg_sequential_opt(x, deterministic=deterministic, x_prior=x_prior, warmup=warmup,
                                             noisy=noisy,
                                             reshape=True, train_prior=train_enc_prior,
                                             num_static_frames=num_static_frames)
        else:
            # static
            x_prior = x_in
            fg_dict = self.fg_module(x_in, deterministic=deterministic, x_prior=x_prior, warmup=warmup, noisy=noisy,
                                     train_prior=train_enc_prior, refinement_iter=True)

        # encoder
        kp_p = fg_dict['kp_p']
        mu = fg_dict['mu']
        logvar = fg_dict['logvar']
        z_base = fg_dict['z_base']
        z = fg_dict['z']
        mu_offset = fg_dict['mu_offset']
        logvar_offset = fg_dict['logvar_offset']
        mu_features = fg_dict['mu_features']
        logvar_features = fg_dict['logvar_features']
        z_features = fg_dict['z_features']
        cropped_objects = fg_dict['cropped_objects_original']
        obj_on_a = fg_dict['obj_on_a']
        obj_on_b = fg_dict['obj_on_b']
        z_obj_on = fg_dict['obj_on']
        mu_depth = fg_dict['mu_depth']
        logvar_depth = fg_dict['logvar_depth']
        z_depth = fg_dict['z_depth']
        mu_scale = fg_dict['mu_scale']
        logvar_scale = fg_dict['logvar_scale']
        z_scale = fg_dict['z_scale']

        # decoder
        bg_mask = fg_dict['bg_mask']
        dec_objects = fg_dict['dec_objects_original']
        dec_objects_trans = fg_dict['dec_objects']
        alpha_masks = fg_dict['alpha_masks']

        if bg_masks_from_fg:
            bg_enc_mask = bg_mask
        else:
            bg_enc_mask = self.get_bg_mask_from_particle_glimpses(z, z_obj_on, mask_size=x_in.shape[-1])
        bg_dict = self.bg_module(x_in, bg_enc_mask, deterministic)
        mu_bg = bg_dict['mu_bg']
        logvar_bg = bg_dict['logvar_bg']
        z_bg = bg_dict['z_bg']
        z_kp_bg = bg_dict['z_kp']
        bg = bg_dict['bg_rec']

        # stitch
        rec = bg_mask * bg + dec_objects_trans

        # dynamics - all but the last timestep
        if forward_dyn:
            # forward PINT
            z_v = z.view(batch_size, timestep_horizon, *z.shape[1:])[:, :-1]
            z_scale_v = z_scale.view(batch_size, timestep_horizon, *z_scale.shape[1:])[:, :-1]
            z_obj_on_v = z_obj_on.view(batch_size, timestep_horizon, *z_obj_on.shape[1:])[:, :-1]
            z_depth_v = z_depth.view(batch_size, timestep_horizon, *z_depth.shape[1:])[:, :-1]
            z_features_v = z_features.view(batch_size, timestep_horizon, *z_features.shape[1:])[:, :-1]
            # [bs, T-1, n_kp, attribute/feature_dim]
            z_bg_features_v = z_bg.view(batch_size, timestep_horizon, *z_bg.shape[1:])[:, :-1]
            # [bs, T-1, bg_learned_feature_dim]

            dyn_out = self.dyn_module(z_v, z_scale_v, z_obj_on_v, z_depth_v, z_features_v, z_bg_features_v)
            mu_dyn = dyn_out['mu']
            logvar_dyn = dyn_out['logvar']

            mu_features_dyn = dyn_out['mu_features']
            logvar_features_dyn = dyn_out['logvar_features']

            obj_on_a_dyn = dyn_out['obj_on_a']
            obj_on_b_dyn = dyn_out['obj_on_b']

            mu_depth_dyn = dyn_out['mu_depth']
            logvar_depth_dyn = dyn_out['logvar_depth']

            mu_scale_dyn = dyn_out['mu_scale']
            logvar_scale_dyn = dyn_out['logvar_scale']

            mu_bg_features_dyn = dyn_out['mu_bg_features']
            logvar_bg_features_dyn = dyn_out['logvar_bg_features']
        else:
            mu_dyn = None
            logvar_dyn = None

            mu_features_dyn = None
            logvar_features_dyn = None

            obj_on_a_dyn = None
            obj_on_b_dyn = None

            mu_depth_dyn = None
            logvar_depth_dyn = None

            mu_scale_dyn = None
            logvar_scale_dyn = None

            mu_bg_features_dyn = None
            logvar_bg_features_dyn = None

        output_dict = {}
        output_dict['kp_p'] = kp_p
        output_dict['rec'] = rec
        output_dict['mu'] = mu
        output_dict['logvar'] = logvar
        output_dict['z_base'] = z_base
        output_dict['z'] = z
        output_dict['z_kp_bg'] = z_kp_bg
        output_dict['mu_offset'] = mu_offset
        output_dict['logvar_offset'] = logvar_offset
        output_dict['mu_features'] = mu_features
        output_dict['logvar_features'] = logvar_features
        output_dict['z_features'] = z_features
        output_dict['bg'] = bg
        output_dict['mu_bg'] = mu_bg
        output_dict['logvar_bg'] = logvar_bg
        output_dict['z_bg'] = z_bg
        # object stuff
        output_dict['cropped_objects_original'] = cropped_objects  # original
        output_dict['obj_on_a'] = obj_on_a
        output_dict['obj_on_b'] = obj_on_b
        output_dict['obj_on'] = z_obj_on
        output_dict['dec_objects_original'] = dec_objects
        output_dict['dec_objects'] = dec_objects_trans
        output_dict['mu_depth'] = mu_depth
        output_dict['logvar_depth'] = logvar_depth
        output_dict['z_depth'] = z_depth
        output_dict['mu_scale'] = mu_scale
        output_dict['logvar_scale'] = logvar_scale
        output_dict['z_scale'] = z_scale
        output_dict['alpha_masks'] = alpha_masks
        # dynamics stuff
        output_dict['mu_dyn'] = mu_dyn
        output_dict['logvar_dyn'] = logvar_dyn
        output_dict['mu_features_dyn'] = mu_features_dyn
        output_dict['logvar_features_dyn'] = logvar_features_dyn
        output_dict['obj_on_a_dyn'] = obj_on_a_dyn
        output_dict['obj_on_b_dyn'] = obj_on_b_dyn
        output_dict['mu_depth_dyn'] = mu_depth_dyn
        output_dict['logvar_depth_dyn'] = logvar_depth_dyn
        output_dict['mu_scale_dyn'] = mu_scale_dyn
        output_dict['logvar_scale_dyn'] = logvar_scale_dyn
        output_dict['mu_bg_dyn'] = mu_bg_features_dyn
        output_dict['logvar_bg_dyn'] = logvar_bg_features_dyn

        return output_dict

    def calc_elbo(self, x, model_output, warmup=False, beta_kl=0.05, beta_dyn=1.0, beta_rec=1.0,
                  kl_balance=0.001, dynamic_discount=None, recon_loss_type="mse", recon_loss_func=None, balance=0.5,
                  beta_dyn_rec=0.1, num_static=4, noisy=False):
        # x: [batch_size, timestep_horizon, ch, h, w]
        # num_static: "burn-in frames" number of timesteps to consider as posterior for the kl with constant priors
        # define losses
        kl_loss_func = ChamferLossKL(use_reverse_kl=False)
        if recon_loss_type == "vgg":
            if recon_loss_func is None:
                recon_loss_func = VGGDistance(device=x.device)
                # recon_loss_func = LPIPS().to(x.device)
        else:
            recon_loss_func = calc_reconstruction_loss

        # unpack output
        mu_p = model_output['kp_p']
        # gmap = model_output['gmap']
        mu = model_output['mu']
        logvar = model_output['logvar']
        z = model_output['z']
        z_base = model_output['z_base']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec']
        mu_features = model_output['mu_features']
        logvar_features = model_output['logvar_features']
        z_features = model_output['z_features']
        mu_bg = model_output['mu_bg']
        logvar_bg = model_output['logvar_bg']
        z_bg = model_output['z_bg']
        mu_scale = model_output['mu_scale']
        logvar_scale = model_output['logvar_scale']
        z_scale = model_output['z_scale']
        mu_depth = model_output['mu_depth']
        logvar_depth = model_output['logvar_depth']
        z_depth = model_output['z_depth']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        cropped_objects_original = model_output['cropped_objects_original']
        obj_on = model_output['obj_on']  # [batch_size, n_kp]
        obj_on_a = model_output['obj_on_a']  # [batch_size, n_kp]
        obj_on_b = model_output['obj_on_b']  # [batch_size, n_kp]
        alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]

        # dynamics stuff
        mu_dyn = model_output['mu_dyn']
        logvar_dyn = model_output['logvar_dyn']
        mu_features_dyn = model_output['mu_features_dyn']
        logvar_features_dyn = model_output['logvar_features_dyn']
        obj_on_a_dyn = model_output['obj_on_a_dyn']
        obj_on_b_dyn = model_output['obj_on_b_dyn']
        mu_depth_dyn = model_output['mu_depth_dyn']
        logvar_depth_dyn = model_output['logvar_depth_dyn']
        mu_scale_dyn = model_output['mu_scale_dyn']
        logvar_scale_dyn = model_output['logvar_scale_dyn']
        mu_bg_features_dyn = model_output['mu_bg_dyn']
        logvar_bg_features_dyn = model_output['logvar_bg_dyn']

        batch_size = x.shape[0]
        timestep_horizon = self.timestep_horizon
        x = x.view(-1, *x.shape[2:])
        static_scale = 1
        dyn_scale = 1
        # discount for future steps
        if dynamic_discount is None:
            discount = torch.ones(size=(timestep_horizon - num_static + 1,), device=x.device)
        else:
            discount = dynamic_discount[:timestep_horizon - num_static + 1]

        # --- reconstruction error --- #
        if dec_objects_original is not None and warmup:
            if recon_loss_type == "vgg":
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.reshape(-1,
                                                                            *cropped_objects_original.shape[2:])
                if cropped_objects_original.shape[-1] < 32:
                    cropped_objects_original = F.interpolate(cropped_objects_original, size=32, mode='bilinear',
                                                             align_corners=False)
                    dec_objects_rgb = F.interpolate(dec_objects_rgb, size=32, mode='bilinear',
                                                    align_corners=False)
                loss_rec_obj = recon_loss_func(cropped_objects_original, dec_objects_rgb, reduction="none")

            else:
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                # [batch_size * timestep_horizon, n_kp_enc, 3, patch_size, patch_size]
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                # [batch_size * timestep_horizon * n_kp_enc, 3, patch_size, patch_size]
                cropped_objects_original = cropped_objects_original.clone().reshape(-1,
                                                                                    *cropped_objects_original.shape[
                                                                                     2:])
                loss_rec_obj = calc_reconstruction_loss(cropped_objects_original, dec_objects_rgb,
                                                        loss_type='mse', reduction='none')
            loss_rec = loss_rec_obj + (0 * rec_x).mean()  # + (0 * rec_x).mean() for distributed training
            loss_rec = loss_rec.view(batch_size, timestep_horizon + 1, -1)
            # [batch_size, timestep_horizon + 1, n_kp_enc]
            loss_rec_0, loss_rec_future = loss_rec.split([num_static, timestep_horizon - num_static + 1], dim=1)
            loss_rec_future = beta_dyn_rec * discount[None, :, None] * loss_rec_future
            loss_rec = loss_rec_0.sum(dim=(-2, -1)).mean() + loss_rec_future.sum(dim=(-2, -1)).mean()
            # loss_rec = static_scale * loss_rec_0.sum(dim=(-2, -1)).mean() + dyn_scale * loss_rec_future.sum(dim=(-2, -1)).mean()
            psnr = torch.tensor(0.0, dtype=torch.float, device=x.device)
        else:
            if recon_loss_type == "vgg":
                loss_rec = recon_loss_func(x, rec_x, reduction="none")
            else:
                loss_rec = calc_reconstruction_loss(x, rec_x, loss_type='mse', reduction='none')

            loss_rec = loss_rec.view(batch_size, timestep_horizon + 1, -1)
            # consider discount
            loss_rec_0, loss_rec_future = loss_rec.split([num_static, timestep_horizon - num_static + 1], dim=1)
            loss_rec_future = beta_dyn_rec * discount[None, :, None] * loss_rec_future
            loss_rec = loss_rec_0.sum(dim=(-2, -1)).mean() + loss_rec_future.sum(dim=(-2, -1)).mean()
            # loss_rec = static_scale * loss_rec_0.sum(dim=(-2, -1)).mean() + dyn_scale * loss_rec_future.sum(dim=(-2, -1)).mean()

            with torch.no_grad():
                psnr = -10 * torch.log10(F.mse_loss(rec_x, x))
        # --- end reconstruction error --- #

        # --- isolate the first timestep for kl with constant priors --- #
        # let num_static = t, timestep_horizon = T
        mu_0 = mu.reshape(batch_size, timestep_horizon + 1, *mu.shape[1:])[:, :num_static]  # [bs, t n_kp, 2]
        logvar_0 = logvar.reshape(batch_size, timestep_horizon + 1, *logvar.shape[1:])[:,
                   :num_static]  # [bs, t, n_kp, 2]
        mu_p_0 = mu_p.reshape(batch_size, timestep_horizon + 1, *mu_p.shape[1:])[:,
                 :num_static]  # [bs, t, n_kp_prior, 2]
        mu_offset_0 = mu_offset.reshape(batch_size, timestep_horizon + 1, *mu_offset.shape[1:])[:,
                      :num_static]  # [bs, t, n_kp, 2]
        logvar_offset_0 = logvar_offset.reshape(batch_size, timestep_horizon + 1, *logvar_offset.shape[1:])[:,
                          :num_static]
        mu_depth_0 = mu_depth.reshape(batch_size, timestep_horizon + 1, *mu_depth.shape[1:])[:,
                     :num_static]  # [bs, n_kp, 1]
        logvar_depth_0 = logvar_depth.reshape(batch_size, timestep_horizon + 1, *logvar_depth.shape[1:])[:, :num_static]
        mu_scale_0 = mu_scale.reshape(batch_size, timestep_horizon + 1, *mu_scale.shape[1:])[:,
                     :num_static]  # [bs, t, n_kp, 2]
        logvar_scale_0 = logvar_scale.reshape(batch_size, timestep_horizon + 1, *logvar_scale.shape[1:])[:, :num_static]
        obj_on_a_0 = obj_on_a.reshape(batch_size, timestep_horizon + 1, *obj_on_a.shape[1:])[:,
                     :num_static]  # [bs, t, n_kp]
        obj_on_b_0 = obj_on_b.reshape(batch_size, timestep_horizon + 1, *obj_on_b.shape[1:])[:,
                     :num_static]  # [bs, t, n_kp]
        mu_features_0 = mu_features.reshape(batch_size, timestep_horizon + 1, *mu_features.shape[1:])[:, :num_static]
        logvar_features_0 = logvar_features.reshape(batch_size, timestep_horizon + 1, *logvar_features.shape[1:])[:,
                            :num_static]
        # mu/logvar_features_0: [bs, n_kp, feat_dim]
        mu_bg_0 = mu_bg.reshape(batch_size, timestep_horizon + 1, *mu_bg.shape[1:])[:, :num_static]  # [bs, t, feat_dim]
        logvar_bg_0 = logvar_bg.reshape(batch_size, timestep_horizon + 1, *logvar_bg.shape[1:])[:, :num_static]
        # note: bg latent dim = a single particle's latent dim

        # --- end isolate the first timestep for kl with constant priors --- #

        # --- define priors --- #
        warmup_logvar = torch.log(torch.tensor(0.1 ** 2))
        logvar_kp = self.logvar_kp.expand_as(mu_p_0)
        logvar_offset_p = warmup_logvar if (warmup or noisy) else self.logvar_offset_p
        logvar_scale_p = warmup_logvar if (warmup or noisy) else self.logvar_scale_p
        # encourage objects to be 'on' during warmup
        obj_on_a_prior = torch.tensor(0.2, device=obj_on_a.device) if (warmup or noisy) else self.obj_on_a_p
        obj_on_b_prior = torch.tensor(0.1, device=obj_on_b.device) if (warmup or noisy) else self.obj_on_b_p
        mu_scale_prior = self.mu_scale_prior
        mu_tot = z_base.detach() + mu_offset  # TODO: detach z_base necessary?
        logvar_tot = logvar_offset
        # --- end priors --- #

        # --- kl-divergence for t <= tau --- #
        # kl-divergence and priors
        mu_prior = mu_p_0.reshape(-1, *mu_p_0.shape[2:])  # [bs * t, n_kp_prior, 2]
        logvar_prior = logvar_kp.reshape(-1, *logvar_kp.shape[2:])  # [bs * t, n_kp_prior, 2]
        # for t < tau, we separate for base and offset
        mu_post = mu_0.reshape(-1, *mu_0.shape[2:])  # [bs * t, n_kp, 2]
        # deterministic chamfer (separable chamfer-kl)
        logvar_post = torch.zeros_like(mu_post)
        # note: mu_p_0 is a duplication of the prior keypoints for the first timestep
        loss_kl_kp_base = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
                                       logvar_gts=logvar_prior)  # [batch_size, ]
        loss_kl_kp_base = loss_kl_kp_base.view(-1, num_static).sum(-1).mean()
        loss_kl_kp_base = loss_kl_kp_base.mean()

        loss_kl_kp_offset = calc_kl(logvar_offset_0.reshape(-1, logvar_offset_0.shape[-1]),
                                    mu_offset_0.reshape(-1, mu_offset_0.shape[-1]), logvar_o=logvar_offset_p,
                                    reduce='none')
        loss_kl_kp_offset = (loss_kl_kp_offset.view(-1, num_static, self.n_kp_enc)).sum(dim=(-2, -1)).mean()
        # loss_kl_kp = loss_kl_kp_base + loss_kl_kp_offset
        loss_kl_kp = 0.5 * kl_balance * loss_kl_kp_base + loss_kl_kp_offset
        # loss_kl_kp = loss_kl_kp_base + kl_balance * loss_kl_kp_offset

        # depth
        loss_kl_depth = calc_kl(logvar_depth_0.reshape(-1, logvar_depth_0.shape[-1]),
                                mu_depth_0.reshape(-1, mu_depth_0.shape[-1]), reduce='none')
        loss_kl_depth = (loss_kl_depth.view(-1, num_static, self.n_kp_enc)).sum(dim=(-2, -1)).mean()

        # scale
        # assume sigmoid activation on z_scale
        loss_kl_scale = calc_kl(logvar_scale_0.reshape(-1, logvar_scale_0.shape[-1]),
                                mu_scale_0.reshape(-1, mu_scale_0.shape[-1]),
                                mu_o=mu_scale_prior, logvar_o=logvar_scale_p,
                                reduce='none')
        loss_kl_scale = (loss_kl_scale.view(-1, num_static, self.n_kp_enc)).sum(dim=(-2, -1)).mean()

        # obj_on (transparency)
        loss_kl_obj_on = calc_kl_beta_dist(obj_on_a_0, obj_on_b_0,
                                           obj_on_a_prior,
                                           obj_on_b_prior).sum(dim=(-2, -1))
        loss_kl_obj_on = loss_kl_obj_on.mean()
        obj_on_l1 = torch.abs(obj_on).sum(-1).mean()  # just to get an idea how many particles are turned on

        # features
        loss_kl_feat = calc_kl(logvar_features_0.reshape(-1, logvar_features_0.shape[-1]),
                               mu_features_0.reshape(-1, mu_features_0.shape[-1]), reduce='none')
        loss_kl_feat_obj = loss_kl_feat.view(-1, num_static, self.n_kp_enc)
        loss_kl_feat_obj = loss_kl_feat_obj.sum(dim=(-2, -1)).mean()

        loss_kl_feat_bg = calc_kl(logvar_bg_0, mu_bg_0, reduce='none')
        loss_kl_feat_bg = loss_kl_feat_bg.mean()
        loss_kl_feat = loss_kl_feat_obj + loss_kl_feat_bg

        # --- end kl-divergence for t < tau --- #

        # --- kl-divergence for t >= tau --- #
        # dynamics
        # transparency
        obj_on_a_post = obj_on_a.reshape(batch_size, timestep_horizon + 1, *obj_on_a.shape[1:])[:, num_static:]
        obj_on_b_post = obj_on_b.reshape(batch_size, timestep_horizon + 1, *obj_on_b.shape[1:])[:, num_static:]

        obj_on_a_post = obj_on_a_post.reshape(-1, *obj_on_a_post.shape[2:])
        obj_on_b_post = obj_on_b_post.reshape(-1, *obj_on_b_post.shape[2:])
        obj_on_a_dyn = obj_on_a_dyn[:, num_static - 1:].reshape(-1, *obj_on_a_dyn.shape[2:])
        obj_on_b_dyn = obj_on_b_dyn[:, num_static - 1:].reshape(-1, *obj_on_b_dyn.shape[2:])

        loss_kl_obj_on_dyn = calc_kl_beta_dist(obj_on_a_post, obj_on_b_post,
                                               obj_on_a_dyn, obj_on_b_dyn, balance=balance).sum(-1)
        loss_kl_obj_on_dyn = loss_kl_obj_on_dyn.reshape(batch_size, -1)
        loss_kl_obj_on_dyn = (loss_kl_obj_on_dyn * discount[None, :]).sum(-1).mean()

        # position
        mu_dyn_post_offset = mu_tot.reshape(batch_size, timestep_horizon + 1, *mu_tot.shape[1:])[:, num_static:]
        logvar_dyn_post_offset = logvar_tot.reshape(batch_size, timestep_horizon + 1, *logvar_tot.shape[1:])[:,
                                 num_static:]
        loss_kl_kp_dyn_offset = calc_kl(logvar_dyn_post_offset.reshape(-1, logvar_dyn_post_offset.shape[-1]),
                                        mu_dyn_post_offset.reshape(-1, mu_dyn_post_offset.shape[-1]),
                                        mu_o=mu_dyn[:, num_static - 1:].reshape(-1, mu_dyn.shape[-1]),
                                        logvar_o=logvar_dyn[:, num_static - 1:].reshape(-1, logvar_dyn.shape[-1]),
                                        reduce='none',
                                        balance=balance)
        loss_kl_kp_dyn = loss_kl_kp_dyn_offset
        loss_kl_kp_dyn = (loss_kl_kp_dyn.view(batch_size, -1, self.n_kp_enc)).sum(-1)
        loss_kl_kp_dyn = (loss_kl_kp_dyn * discount[None, :]).sum(-1).mean()

        # scale
        mu_scale_dyn_post = mu_scale.reshape(batch_size, timestep_horizon + 1, *mu_scale.shape[1:])[:, num_static:]
        logvar_scale_dyn_post = logvar_scale.reshape(batch_size, timestep_horizon + 1, *logvar_scale.shape[1:])[:,
                                num_static:]
        loss_kl_scale_dyn = calc_kl(logvar_scale_dyn_post.reshape(-1, logvar_scale_dyn_post.shape[-1]),
                                    mu_scale_dyn_post.reshape(-1, mu_scale_dyn_post.shape[-1]),
                                    mu_o=mu_scale_dyn[:, num_static - 1:].reshape(-1, mu_scale_dyn.shape[-1]),
                                    logvar_o=logvar_scale_dyn[:, num_static - 1:].reshape(-1,
                                                                                          logvar_scale_dyn.shape[-1]),
                                    reduce='none', balance=balance)
        loss_kl_scale_dyn = (loss_kl_scale_dyn.view(batch_size, -1, self.n_kp_enc)).sum(-1)
        loss_kl_scale_dyn = (loss_kl_scale_dyn * discount[None, :]).sum(-1).mean()

        # depth
        mu_depth_dyn_post = mu_depth.reshape(batch_size, timestep_horizon + 1, *mu_depth.shape[1:])[:, num_static:]
        logvar_depth_dyn_post = logvar_depth.reshape(batch_size, timestep_horizon + 1, *logvar_depth.shape[1:])[:,
                                num_static:]
        loss_kl_depth_dyn = calc_kl(logvar_depth_dyn_post.reshape(-1, logvar_depth_dyn_post.shape[-1]),
                                    mu_depth_dyn_post.reshape(-1, mu_depth_dyn_post.shape[-1]),
                                    mu_o=mu_depth_dyn[:, num_static - 1:].reshape(-1, mu_depth_dyn.shape[-1]),
                                    logvar_o=logvar_depth_dyn[:, num_static - 1:].reshape(-1,
                                                                                          logvar_depth_dyn.shape[-1]),
                                    reduce='none', balance=balance)
        loss_kl_depth_dyn = (loss_kl_depth_dyn.view(batch_size, -1, self.n_kp_enc)).sum(-1)
        loss_kl_depth_dyn = (loss_kl_depth_dyn * discount[None, :]).sum(-1).mean()

        # features
        mu_features_dyn_post = mu_features.reshape(batch_size, timestep_horizon + 1, *mu_features.shape[1:])[:,
                               num_static:]
        logvar_features_dyn_post = logvar_features.reshape(batch_size, timestep_horizon + 1,
                                                           *logvar_features.shape[1:])[:, num_static:]
        loss_kl_features_dyn = calc_kl(logvar_features_dyn_post.reshape(-1, logvar_features_dyn_post.shape[-1]),
                                       mu_features_dyn_post.reshape(-1, mu_features_dyn_post.shape[-1]),
                                       mu_o=mu_features_dyn[:, num_static - 1:].reshape(-1, mu_features_dyn.shape[-1]),
                                       logvar_o=logvar_features_dyn[:, num_static - 1:].reshape(-1,
                                                                                                logvar_features_dyn.shape[
                                                                                                    -1]),
                                       reduce='none', balance=balance)
        loss_kl_features_dyn = (
            loss_kl_features_dyn.view(batch_size, -1, self.n_kp_enc)).sum(-1)
        loss_kl_features_dyn = (loss_kl_features_dyn * discount[None, :]).sum(-1).mean()

        # bg featuers
        mu_bg_dyn_post = mu_bg.reshape(batch_size, timestep_horizon + 1, *mu_bg.shape[1:])[:, num_static:]
        logvar_bg_dyn_post = logvar_bg.reshape(batch_size, timestep_horizon + 1, *logvar_bg.shape[1:])[:, num_static:]
        loss_kl_bg_dyn = calc_kl(logvar_bg_dyn_post.reshape(-1, logvar_bg_dyn_post.shape[-1]),
                                 mu_bg_dyn_post.reshape(-1, mu_bg_dyn_post.shape[-1]),
                                 mu_o=mu_bg_features_dyn[:, num_static - 1:].reshape(-1, mu_bg_features_dyn.shape[-1]),
                                 logvar_o=logvar_bg_features_dyn[:, num_static - 1:].reshape(-1,
                                                                                             logvar_bg_features_dyn.shape[
                                                                                                 -1]),
                                 reduce='none', balance=balance)
        loss_kl_bg_dyn = loss_kl_bg_dyn.view(batch_size, -1)
        loss_kl_bg_dyn = (loss_kl_bg_dyn * discount[None, :]).sum(-1).mean()

        # total dynamics kl
        loss_kl_dyn = sum([loss_kl_kp_dyn, loss_kl_obj_on_dyn, loss_kl_depth_dyn, loss_kl_scale_dyn,
                           loss_kl_features_dyn, loss_kl_bg_dyn])

        # --- end kl-divergence for t >= tau --- #

        # total losses
        # loss_kl = loss_kl_kp + loss_kl_depth + loss_kl_scale + loss_kl_obj_on + kl_balance * loss_kl_feat
        loss_kl = loss_kl_kp + loss_kl_scale + loss_kl_obj_on + kl_balance * (loss_kl_feat + loss_kl_depth)
        # loss_kl = kl_balance * loss_kl_kp + loss_kl_depth + loss_kl_scale + loss_kl_obj_on + kl_balance * loss_kl_feat
        # loss_kl = kl_balance * (loss_kl_kp + loss_kl_depth + loss_kl_scale + loss_kl_obj_on + loss_kl_feat)
        # loss = (1 / (timestep_horizon + 1)) * (beta_rec * loss_rec + beta_kl * loss_kl + beta_dyn * loss_kl_dyn)
        loss = (1 / (timestep_horizon + 1)) * (
                beta_rec * loss_rec + beta_kl * static_scale * loss_kl + beta_dyn * dyn_scale * loss_kl_dyn)
        # loss = beta_rec * loss_rec + beta_kl * static_scale * loss_kl + beta_dyn * dyn_scale * loss_kl_dyn
        if recon_loss_type == "vgg":
            loss = 1e-3 * loss
            # scale-by-pixel
            # loss_scale = 1 / (x.shape[1] * x.shape[2] * x.shape[3])
            # loss = 1e-7 * loss
        loss_dict = {'loss': loss, 'psnr': psnr.detach(), 'kl': loss_kl, 'kl_dyn': loss_kl_dyn, 'loss_rec': loss_rec,
                     'obj_on_l1': obj_on_l1, 'loss_kl_kp': loss_kl_kp, 'loss_kl_feat': loss_kl_feat,
                     'loss_kl_obj_on': loss_kl_obj_on, 'loss_kl_scale': loss_kl_scale, 'loss_kl_depth': loss_kl_depth}
        return loss_dict

    def calc_elbo_static(self, x, model_output, warmup=False, beta_kl=0.05, beta_dyn=1.0, beta_rec=1.0,
                         kl_balance=0.001, dynamic_discount=None, recon_loss_type="mse", recon_loss_func=None,
                         balance=0.5, noisy=False):
        # x: [batch_size, timestep_horizon, ch, h, w]
        # constant prior for all timesteps (single image DLP)
        # balance: kl balance for dynamics kl posterior and prior
        # define losses
        kl_loss_func = ChamferLossKL(use_reverse_kl=False)
        if recon_loss_type == "vgg":
            if recon_loss_func is None:
                recon_loss_func = VGGDistance(device=x.device)
                # recon_loss_func = LPIPS().to(x.device)
        else:
            recon_loss_func = calc_reconstruction_loss

        # unpack output
        mu_p = model_output['kp_p']
        # gmap = model_output['gmap']
        mu = model_output['mu']
        logvar = model_output['logvar']
        z = model_output['z']
        z_base = model_output['z_base']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec']
        mu_features = model_output['mu_features']
        logvar_features = model_output['logvar_features']
        z_features = model_output['z_features']
        mu_bg = model_output['mu_bg']
        logvar_bg = model_output['logvar_bg']
        z_bg = model_output['z_bg']
        mu_scale = model_output['mu_scale']
        logvar_scale = model_output['logvar_scale']
        z_scale = model_output['z_scale']
        mu_depth = model_output['mu_depth']
        logvar_depth = model_output['logvar_depth']
        z_depth = model_output['z_depth']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        cropped_objects_original = model_output['cropped_objects_original']
        obj_on = model_output['obj_on']  # [batch_size, n_kp]
        obj_on_a = model_output['obj_on_a']  # [batch_size, n_kp]
        obj_on_b = model_output['obj_on_b']  # [batch_size, n_kp]
        alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]

        batch_size = x.shape[0]
        timestep_horizon = self.timestep_horizon
        x = x.view(-1, *x.shape[2:])

        # --- reconstruction error --- #
        if dec_objects_original is not None and warmup:
            if recon_loss_type == "vgg":
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.reshape(-1,
                                                                            *cropped_objects_original.shape[2:])
                if cropped_objects_original.shape[-1] < 32:
                    cropped_objects_original = F.interpolate(cropped_objects_original, size=32, mode='bilinear',
                                                             align_corners=False)
                    dec_objects_rgb = F.interpolate(dec_objects_rgb, size=32, mode='bilinear',
                                                    align_corners=False)
                loss_rec_obj = recon_loss_func(cropped_objects_original, dec_objects_rgb, reduction="mean")

            else:
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.clone().reshape(-1,
                                                                                    *cropped_objects_original.shape[
                                                                                     2:])
                loss_rec_obj = calc_reconstruction_loss(cropped_objects_original, dec_objects_rgb,
                                                        loss_type='mse', reduction='mean')
            loss_rec = loss_rec_obj + (0 * rec_x).mean()  # + (0 * rec_x).mean() for distributed training
            psnr = torch.tensor(0.0, dtype=torch.float, device=x.device)
        else:
            if recon_loss_type == "vgg":
                loss_rec = recon_loss_func(x, rec_x, reduction="none")
            else:
                loss_rec = calc_reconstruction_loss(x, rec_x, loss_type='mse', reduction='none')

            with torch.no_grad():
                psnr = -10 * torch.log10(F.mse_loss(rec_x, x))
                # batch_psnrs.append(psnr.data.cpu().item())
            loss_rec = loss_rec.view(batch_size, timestep_horizon + 1, -1)
            loss_rec = loss_rec.sum(-1).mean()

        # --- end reconstruction error --- #

        # --- define priors --- #
        warmup_logvar = torch.log(torch.tensor(0.1 ** 2))
        logvar_kp = self.logvar_kp.expand_as(mu_p)
        logvar_offset_p = warmup_logvar if (warmup or noisy) else self.logvar_offset_p
        logvar_scale_p = warmup_logvar if (warmup or noisy) else self.logvar_scale_p
        # encourage objects to be 'on' during warmup
        obj_on_a_prior = torch.tensor(0.2, device=obj_on_a.device) if (warmup or noisy) else self.obj_on_a_p
        obj_on_b_prior = torch.tensor(0.1, device=obj_on_b.device) if (warmup or noisy) else self.obj_on_b_p
        mu_scale_prior = self.mu_scale_prior
        # --- end priors --- #

        # --- kl-divergence for t = 0 --- #
        # kl-divergence and priors
        mu_prior = mu_p
        logvar_prior = logvar_kp
        mu_post = mu
        logvar_post = torch.zeros_like(mu)
        loss_kl_kp_base = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
                                       logvar_gts=logvar_prior)  # [batch_size, ]
        loss_kl_kp_base = loss_kl_kp_base.mean()

        loss_kl_kp_offset = calc_kl(logvar_offset.reshape(-1, logvar_offset.shape[-1]),
                                    mu_offset.reshape(-1, mu_offset.shape[-1]), logvar_o=logvar_offset_p,
                                    reduce='none')
        loss_kl_kp_offset = (loss_kl_kp_offset.view(-1, self.n_kp_enc)).sum(-1).mean()
        loss_kl_kp = 0.5 * kl_balance * loss_kl_kp_base + loss_kl_kp_offset

        # depth
        loss_kl_depth = calc_kl(logvar_depth.reshape(-1, logvar_depth.shape[-1]),
                                mu_depth.reshape(-1, mu_depth.shape[-1]), reduce='none')
        loss_kl_depth = (loss_kl_depth.view(-1, self.n_kp_enc)).sum(-1).mean()

        # scale
        # assume sigmoid activation on z_scale
        loss_kl_scale = calc_kl(logvar_scale.reshape(-1, logvar_scale.shape[-1]),
                                mu_scale.reshape(-1, mu_scale.shape[-1]),
                                mu_o=mu_scale_prior, logvar_o=logvar_scale_p,
                                reduce='none')
        loss_kl_scale = (loss_kl_scale.view(-1, self.n_kp_enc)).sum(-1).mean()

        # obj_on
        loss_kl_obj_on = calc_kl_beta_dist(obj_on_a, obj_on_b,
                                           obj_on_a_prior,
                                           obj_on_b_prior).sum(-1)
        loss_kl_obj_on = loss_kl_obj_on.mean()
        obj_on_l1 = torch.abs(obj_on).sum(-1).mean()  # just to get an idea how many particles are turned on

        # features
        loss_kl_feat = calc_kl(logvar_features.reshape(-1, logvar_features.shape[-1]),
                               mu_features.reshape(-1, mu_features.shape[-1]), reduce='none')
        loss_kl_feat_obj = loss_kl_feat.view(-1, self.n_kp_enc)
        loss_kl_feat_obj = loss_kl_feat_obj.sum(-1).mean()

        loss_kl_feat_bg = calc_kl(logvar_bg, mu_bg, reduce='none')
        loss_kl_feat_bg = loss_kl_feat_bg.mean()
        loss_kl_feat = loss_kl_feat_obj + loss_kl_feat_bg

        # total losses
        # loss_kl = loss_kl_kp + loss_kl_depth + loss_kl_scale + loss_kl_obj_on + kl_balance * loss_kl_feat
        loss_kl = loss_kl_kp + loss_kl_scale + loss_kl_obj_on + kl_balance * (loss_kl_feat + loss_kl_depth)
        loss = beta_rec * loss_rec + beta_kl * loss_kl
        if recon_loss_type == "vgg":
            loss = 1e-3 * loss
        loss_kl_dyn = torch.tensor(0.0, device=x.device)
        loss_dict = {'loss': loss, 'psnr': psnr.detach(), 'kl': loss_kl, 'kl_dyn': loss_kl_dyn, 'loss_rec': loss_rec,
                     'obj_on_l1': obj_on_l1, 'loss_kl_kp': loss_kl_kp, 'loss_kl_feat': loss_kl_feat,
                     'loss_kl_obj_on': loss_kl_obj_on, 'loss_kl_scale': loss_kl_scale, 'loss_kl_depth': loss_kl_depth}
        return loss_dict

    def lerp(self, other, betta):
        # weight interpolation for ema - not used in the paper
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = self.parameters()
            other_param = other.parameters()
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)
