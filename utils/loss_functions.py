"""
Loss functions implementations used in the optimization of DLP.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# functions
def batch_pairwise_kl(mu_x, logvar_x, mu_y, logvar_y, reverse_kl=False):
    """
    Calculate batch-wise KL-divergence
    mu_x, logvar_x: [batch_size, n_x, points_dim]
    mu_y, logvar_y: [batch_size, n_y, points_dim]
    kl = -0.5 * Σ_points_dim (1 + logvar_x - logvar_y - exp(logvar_x)/exp(logvar_y)
                    - ((mu_x - mu_y) ** 2)/exp(logvar_y))
    """
    if reverse_kl:
        mu_a, logvar_a = mu_y, logvar_y
        mu_b, logvar_b = mu_x, logvar_x
    else:
        mu_a, logvar_a = mu_x, logvar_x
        mu_b, logvar_b = mu_y, logvar_y
    bs, n_a, points_dim = mu_a.size()
    _, n_b, _ = mu_b.size()
    logvar_aa = logvar_a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    logvar_bb = logvar_b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    mu_aa = mu_a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    mu_bb = mu_b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    p_kl = -0.5 * (1 + logvar_aa - logvar_bb - logvar_aa.exp() / logvar_bb.exp()
                   - ((mu_aa - mu_bb) ** 2) / logvar_bb.exp()).sum(-1)  # [batch_size, n_x, n_y]
    return p_kl


def batch_pairwise_dist(x, y, metric='l2'):
    assert metric in ['l2', 'l2_simple', 'l1', 'cosine'], f'metric {metric} unrecognized'
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    if metric == 'cosine':
        dist_func = torch.nn.functional.cosine_similarity
        P = -dist_func(x.unsqueeze(2), y.unsqueeze(1), dim=-1, eps=1e-8)
    elif metric == 'l1':
        P = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(-1)
    elif metric == 'l2_simple':
        P = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    else:
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum', balance=0.5):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :param balance: balancing coefficient between posterior and prior
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    if balance == 0.5:
        kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
            logvar_o)).sum(1)
    else:
        # detach post
        mu_post = mu.detach()
        logvar_post = logvar.detach()
        mu_prior = mu_o
        logvar_prior = logvar_o
        kl_a = -0.5 * (1 + logvar_post - logvar_prior - logvar_post.exp() / torch.exp(logvar_prior) - (
                mu_post - mu_prior).pow(2) / torch.exp(logvar_prior)).sum(1)
        # detach prior
        mu_post = mu
        logvar_post = logvar
        mu_prior = mu_o.detach()
        logvar_prior = logvar_o.detach()
        kl_b = -0.5 * (1 + logvar_post - logvar_prior - logvar_post.exp() / torch.exp(logvar_prior) - (
                mu_post - mu_prior).pow(2) / torch.exp(logvar_prior)).sum(1)
        kl = (1 - balance) * kl_a + balance * kl_b
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def calc_kl_bern(post_prob, prior_prob, eps=1e-15, reduce='none'):
    """
    Compute kl divergence of Bernoulli variable
    :param post_prob [batch_size, 1], in [0,1]
    :param prior_prob [batch_size, 1], in [0,1]
    :return: kl divergence, (B, ...)
    """
    kl = post_prob * (torch.log(post_prob + eps) - torch.log(prior_prob + eps)) + (1 - post_prob) * (
            torch.log(1 - post_prob + eps) - torch.log(1 - prior_prob + eps))
    if reduce == 'sum':
        kl = kl.sum()
    elif reduce == 'mean':
        kl = kl.mean()
    else:
        kl = kl.squeeze(-1)
    return kl


def log_beta_function(alpha, beta, eps=1e-5):
    """
    B(alpha, beta) = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    logB = loggamma(alpha) + loggamma(beta) - loggamaa(alpha + beta)
    """
    # return torch.special.gammaln(alpha) + torch.special.gammaln(beta) - torch.special.gammaln(alpha + beta)
    return torch.lgamma(alpha + eps) + torch.lgamma(beta + eps) - torch.lgamma(alpha + beta + eps)


def calc_kl_beta_dist(alpha_post, beta_post, alpha_prior, beta_prior, reduce='none', eps=1e-5, balance=0.5):
    """
    Compute kl divergence of Beta variable
    https://en.wikipedia.org/wiki/Beta_distribution
    :param alpha_post, beta_post [batch_size, 1]
    :param alpha_prior,  beta_prior  [batch_size, 1]
    :param balance kl balance between posterior and prior
    :return: kl divergence, (B, ...)
    """
    if balance == 0.5:
        log_bettas = log_beta_function(alpha_prior, beta_prior) - log_beta_function(alpha_post, beta_post)
        alpha = (alpha_post - alpha_prior) * torch.digamma(alpha_post + eps)
        beta = (beta_post - beta_prior) * torch.digamma(beta_post + eps)
        alpha_beta = (alpha_prior - alpha_post + beta_prior - beta_post) * torch.digamma(alpha_post + beta_post + eps)
        kl = log_bettas + alpha + beta + alpha_beta
    else:
        # detach post
        log_bettas = log_beta_function(alpha_prior, beta_prior) - log_beta_function(alpha_post.detach(),
                                                                                    beta_post.detach())
        alpha = (alpha_post - alpha_prior) * torch.digamma(alpha_post.detach() + eps)
        beta = (beta_post.detach() - beta_prior) * torch.digamma(beta_post.detach() + eps)
        alpha_beta = (alpha_prior - alpha_post.detach() + beta_prior - beta_post.detach()) * torch.digamma(
            alpha_post.detach() + beta_post.detach() + eps)
        kl_a = log_bettas + alpha + beta + alpha_beta

        # detach prior
        log_bettas = log_beta_function(alpha_prior.detach(), beta_prior.detach()) - log_beta_function(alpha_post,
                                                                                                      beta_post)
        alpha = (alpha_post - alpha_prior.detach()) * torch.digamma(alpha_post + eps)
        beta = (beta_post - beta_prior.detach()) * torch.digamma(beta_post + eps)
        alpha_beta = (alpha_prior.detach() - alpha_post + beta_prior.detach() - beta_post) * torch.digamma(
            alpha_post + beta_post + eps)
        kl_b = log_bettas + alpha + beta + alpha_beta
        kl = (1 - balance) * kl_a + balance * kl_b
    if reduce == 'sum':
        kl = kl.sum()
    elif reduce == 'mean':
        kl = kl.mean()
    else:
        kl = kl.squeeze(-1)
    return kl


# classes
class ChamferLossKL(nn.Module):
    """
    Calculates the KL-divergence between two sets of (R.V.) particle coordinates.
    """

    def __init__(self, use_reverse_kl=False):
        super(ChamferLossKL, self).__init__()
        self.use_reverse_kl = use_reverse_kl

    def forward(self, mu_preds, logvar_preds, mu_gts, logvar_gts, posterior_mask=None):
        """
        mu_preds, logvar_preds: [bs, n_x, feat_dim]
        mu_gts, logvar_gts: [bs, n_y, feat_dim]
        posterior_mask: [bs, n_x]
        """
        p_kl = batch_pairwise_kl(mu_preds, logvar_preds, mu_gts, logvar_gts, reverse_kl=False)
        # [bs, n_x, n_y]
        if self.use_reverse_kl:
            p_rkl = batch_pairwise_kl(mu_preds, logvar_preds, mu_gts, logvar_gts, reverse_kl=True)
            p_kl = 0.5 * (p_kl + p_rkl.transpose(2, 1))
        mins, _ = torch.min(p_kl, 1)  # [bs, n_y]
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(p_kl, 2)  # [bs, n_x]
        if posterior_mask is not None:
            mins = mins * posterior_mask
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2


class NetVGGFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = models.vgg16(pretrained=True)
        self.vggnet.eval()
        self.vggnet.requires_grad_(False)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGDistance(nn.Module):

    def __init__(self, layer_ids=(2, 7, 12, 21, 30), accumulate_mode='sum', device=torch.device("cpu"),
                 normalize=True, use_loss_scale=False, vgg_coeff=0.12151):
        super().__init__()

        self.vgg = NetVGGFeatures(layer_ids).to(device)
        self.layer_ids = layer_ids
        self.accumulate_mode = accumulate_mode
        self.device = device
        self.use_normalization = normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.use_loss_scale = use_loss_scale
        self.vgg_coeff = vgg_coeff

    def forward(self, I1, I2, reduction='sum', only_image=False):
        b_sz = I1.size(0)
        num_ch = I1.size(1)

        if self.accumulate_mode == 'sum':
            loss = ((I1 - I2) ** 2).view(b_sz, -1).sum(1)
            # if normalized, effectively: (1 / (std ** 2)) * (I_1 - I_2) ** 2
        elif self.accumulate_mode == 'ch_mean':
            loss = ((I1 - I2) ** 2).view(b_sz, I1.shape[1], -1).mean(1).sum(-1)
        else:
            loss = ((I1 - I2) ** 2).view(b_sz, -1).mean(1)

        if self.use_normalization:
            I1, I2 = self.normalize(I1), self.normalize(I2)

        if num_ch == 1:
            I1 = I1.repeat(1, 3, 1, 1)
            I2 = I2.repeat(1, 3, 1, 1)

        f1 = self.vgg(I1)
        f2 = self.vgg(I2)

        if not only_image:
            for i in range(len(self.layer_ids)):
                if self.accumulate_mode == 'sum':
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, -1).sum(1)
                elif self.accumulate_mode == 'ch_mean':
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, f1[i].shape[1], -1).mean(1).sum(-1)
                else:
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, -1).mean(1)
                c = self.vgg_coeff if self.use_normalization else 1.0
                loss = loss + c * layer_loss

        if self.use_loss_scale:
            # by using `sum` for the features, and using scaling instead of `mean` we maintain the weight
            # of each dimension contribution to the loss
            max_dim = max([np.product(f.shape[1:]) for f in f1])
            scale = 1 / max_dim
            loss = scale * loss
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_dimensions(self, device=torch.device("cpu")):
        dims = []
        dummy_input = torch.zeros(1, 3, 128, 128).to(device)
        dims.append(dummy_input.view(1, -1).size(1))
        f = self.vgg(dummy_input)
        for i in range(len(self.layer_ids)):
            dims.append(f[i].view(1, -1).size(1))
        return dims


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        # self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device, dtype=torch.long)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device, dtype=torch.long)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P


if __name__ == '__main__':
    bs = 32
    n_points_x = 10
    n_points_y = 15
    dim = 8
    x = torch.randn(bs, n_points_x, dim)
    y = torch.randn(bs, n_points_y, dim)
    for metric in ['cosine', 'l1', 'l2', 'l2_simple']:
        P = batch_pairwise_dist(x, y, metric)
        print(f'metric: {metric}, P: {P.shape}, max: {P.max()}, min: {P.min()}')
