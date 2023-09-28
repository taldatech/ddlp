"""
Particle Interaction Transformer (PINT) implementation.
Basic transformer modules adapted from the great minGPT: https://github.com/karpathy/minGPT
"""

# imports
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Beta
from utils.util_func import reparameterize


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
        # seq_len * 3 - 1
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
                 positional_bias=False, max_particles=None, linear_bias=False):
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
                bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
                bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
                att = att + bias_t + bias_p
            else:
                bias_t, _ = self.rel_pos_bias(T)
                bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                att = att + bias_t
        att = att.masked_fill(self.mask[:, :, :, :T, :, :T] == 0, float('-inf'))
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


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None):
        super().__init__()
        self.max_particles = max_particles
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = CausalParticleSelfAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                                positional_bias=positional_bias, max_particles=max_particles)
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ParticleTransformer(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, output_dim, attn_pdrop=0.1, resid_pdrop=0.1,
                 hidden_dim_multiplier=4, positional_bias=False, activation='gelu', max_particles=None):
        super().__init__()
        self.positional_bias = positional_bias
        self.max_particles = max_particles  # for positional bias
        # input embedding stem
        if self.positional_bias:
            self.pos_emb = nn.Identity()
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embed))
        # transformer
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_size, attn_pdrop,
                                            resid_pdrop, hidden_dim_multiplier,
                                            positional_bias, activation=activation, max_particles=max_particles)
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
        # print(f"particle transformer # parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
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

        if not self.positional_bias:
            position_embeddings = self.pos_emb[:, None, :t, :]
            x = x + position_embeddings
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


class ParticleFeatureProjection(torch.nn.Module):
    def __init__(self, in_features_dim, bg_features_dim, hidden_dim, output_dim):
        super().__init__()
        # a projection module to match PINT's inner dimension
        # particles: [z, z_scale, z_obj_on, z_depth, z_features]
        # bg: [z_features]
        self.in_features_dim = in_features_dim
        self.bg_features_dim = bg_features_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.particle_dim = 2 + 2 + 1 + 1 + in_features_dim
        # [z, z_scale, z_obj_on, z_depth, z_features]

        self.xy_projection = nn.Sequential(nn.Linear(2, hidden_dim),
                                           nn.ReLU(True),
                                           nn.Linear(hidden_dim, 2))
        self.scale_projection = nn.Sequential(nn.Linear(2, hidden_dim),
                                              nn.ReLU(True),
                                              nn.Linear(hidden_dim, 2))
        self.obj_on_projection = nn.Sequential(nn.Linear(1, hidden_dim),
                                               nn.ReLU(True),
                                               nn.Linear(hidden_dim, 1))
        self.depth_projection = nn.Sequential(nn.Linear(1, hidden_dim),
                                              nn.ReLU(True),
                                              nn.Linear(hidden_dim, 1))
        self.features_projection = nn.Sequential(nn.Linear(in_features_dim, hidden_dim),
                                                 nn.ReLU(True),
                                                 nn.Linear(hidden_dim, in_features_dim))
        self.particle_projection = nn.Sequential(nn.Linear(self.particle_dim, hidden_dim),
                                                 nn.ReLU(True),
                                                 nn.Linear(hidden_dim, output_dim))
        self.bg_features_projection = nn.Sequential(nn.Linear(bg_features_dim, hidden_dim),
                                                    nn.ReLU(True),
                                                    nn.Linear(hidden_dim, output_dim))

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features):
        # z, z_scale, z_velocity: [bs, n_particles, 2]
        # z_depth, z_obj_on: [bs, n_particles, 1]
        # z_features: [bs, n_particles, in_features_dim]
        # z_bg_features: [bs, bg_features_dim]
        bs, n_particles, feat_dim = z_features.shape
        z_proj = self.xy_projection(z)
        z_scale_proj = self.scale_projection(z_scale)
        if len(z_obj_on.shape) == 2:
            z_obj_on = z_obj_on.unsqueeze(-1)
        z_obj_on_proj = self.obj_on_projection(z_obj_on)
        z_depth_proj = self.depth_projection(z_depth)
        z_features_proj = self.features_projection(z_features)
        z_all = torch.cat([z_proj, z_scale_proj, z_obj_on_proj, z_depth_proj, z_features_proj], dim=-1)
        # z_all: [bs, n_particles, 2 + 2 + 1 + 1 + in_features_dim]
        z_all_proj = self.particle_projection(z_all)  # [bs, n_particles, output_dim]  or [bs, n_particle, hidden_dim]
        z_bg_features_proj = self.bg_features_projection(z_bg_features)  # [bs, output_dim] or [bs, hidden_dim]
        # concat
        z_processed = torch.cat([z_all_proj, z_bg_features_proj.unsqueeze(1)], dim=1)
        # [bs, n_particles + 1, output_dim] or [bs, n_particles + 1, hidden_dim]
        return z_processed


class ParticleFeatureDecoder(nn.Module):
    def __init__(self, input_dim, features_dim, bg_features_dim, hidden_dim, kp_activation='tanh', max_delta=1.0,
                 delta_features=False):
        super().__init__()
        # decoder to map back from PINT's inner dim to the paricle's original dimension
        self.input_dim = input_dim
        self.features_dim = features_dim
        self.bg_features_dim = bg_features_dim
        self.kp_activation = kp_activation
        self.max_delta = max_delta
        self.delta_features = delta_features
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.ReLU(True),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(True)
                                      )
        self.x_head = nn.Linear(hidden_dim, 2)  # mu_x, logvar_x
        self.y_head = nn.Linear(hidden_dim, 2)  # mu_y, logvar_y
        self.scale_xy_head = nn.Linear(hidden_dim, 4)  # mu_sx, logvar_sx, mu_sy, logvar_sy
        self.obj_on_head = nn.Linear(hidden_dim, 2)  # log_a, log_b
        self.depth_head = nn.Linear(hidden_dim, 2)  # mu_z, logvar_z
        self.features_head = nn.Linear(hidden_dim, 2 * features_dim)  # mu_features, logvar_features
        self.bg_features_head = nn.Linear(hidden_dim, 2 * bg_features_dim)  # mu_features, logvar_features

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        # x: [bs, n_particles + 1, input_dim]
        bs, n_particles, in_dim = x.shape
        backbone_features = self.backbone(x)
        fg_features, bg_features = backbone_features.split([n_particles - 1, 1], dim=1)
        stats_x = self.x_head(fg_features)
        stats_x = stats_x.view(bs, n_particles - 1, 2)
        mu_x, logvar_x = stats_x.chunk(chunks=2, dim=-1)

        stats_y = self.y_head(fg_features)
        stats_y = stats_y.view(bs, n_particles - 1, 2)
        mu_y, logvar_y = stats_y.chunk(chunks=2, dim=-1)

        mu = torch.cat([mu_x, mu_y], dim=-1)
        logvar = torch.cat([logvar_x, logvar_y], dim=-1)

        if self.kp_activation == "tanh":
            mu = torch.tanh(mu)
        elif self.kp_activation == "sigmoid":
            mu = torch.sigmoid(mu)

        # apply max delta
        mu = self.max_delta * mu

        scale_xy = self.scale_xy_head(fg_features)
        scale_xy = scale_xy.view(bs, n_particles - 1, -1)
        mu_scale, logvar_scale = torch.chunk(scale_xy, chunks=2, dim=-1)

        obj_on = self.obj_on_head(fg_features)
        obj_on = obj_on.view(bs, n_particles - 1, 2)
        lobj_on_a, lobj_on_b = torch.chunk(obj_on, chunks=2, dim=-1)  # log alpha, beta of Beta dist

        depth = self.depth_head(fg_features)
        depth = depth.view(bs, n_particles - 1, 2)
        mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)

        features = self.features_head(fg_features)
        features = features.view(bs, n_particles - 1, 2 * self.features_dim)
        mu_features, logvar_features = torch.chunk(features, 2, dim=-1)

        bg_features = self.bg_features_head(bg_features.squeeze(1))
        mu_bg_features, logvar_bg_features = torch.chunk(bg_features, 2, dim=-1)

        decoder_out = {'mu': mu, 'logvar': logvar, 'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b,
                       'obj_on': obj_on, 'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'mu_features': mu_features,
                       'logvar_features': logvar_features, 'mu_bg_features': mu_bg_features,
                       'logvar_bg_features': logvar_bg_features}

        return decoder_out


class DynamicsDLP(nn.Module):
    def __init__(self, features_dim, bg_features_dim, hidden_dim, projection_dim,
                 n_head=4, n_layer=2, block_size=12, dropout=0.1,
                 kp_activation='tanh', predict_delta=True, max_delta=1.0,
                 positional_bias=True, max_particles=None):
        super(DynamicsDLP, self).__init__()
        """
        DLP Dynamics Module (PINT)
        """
        self.predict_delta = predict_delta
        self.projection_dim = projection_dim
        self.max_delta = max_delta
        self.max_particles = max_particles  # for positional bias
        self.particle_projection = ParticleFeatureProjection(features_dim, bg_features_dim,
                                                             hidden_dim, self.projection_dim)
        self.particle_transformer = ParticleTransformer(self.projection_dim, n_head, n_layer,
                                                        block_size, self.projection_dim,
                                                        attn_pdrop=dropout, resid_pdrop=dropout,
                                                        hidden_dim_multiplier=4,
                                                        positional_bias=positional_bias,
                                                        activation='gelu', max_particles=max_particles)
        self.particle_decoder = ParticleFeatureDecoder(self.projection_dim, features_dim, bg_features_dim,
                                                       hidden_dim, kp_activation=kp_activation, max_delta=max_delta,
                                                       delta_features=predict_delta)

    def sample(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, steps=10, deterministic=False):
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        """
        block_size = self.particle_transformer.get_block_size()
        # z, z_scale: [bs, T, n_particles, 2]
        # z_depth, z_obj_on: [bs, T, n_particles, 1]
        # z_features: [bs, T, n_particles, in_features_dim]
        # z_bg_features: [bs, T, bg_features_dim]

        bs, timestep_horizon, n_particles, _ = z.shape
        for k in range(steps):
            # project particles
            z_v = z[:, -block_size:].reshape(-1, *z.shape[2:])
            z_scale_v = z_scale[:, -block_size:].reshape(-1, *z_scale.shape[2:])
            z_obj_on_v = z_obj_on[:, -block_size:].reshape(-1, *z_obj_on.shape[2:])
            z_depth_v = z_depth[:, -block_size:].reshape(-1, *z_depth.shape[2:])
            z_features_v = z_features[:, -block_size:].reshape(-1, *z_features.shape[2:])
            z_bg_features_v = z_bg_features[:, -block_size:].reshape(-1, *z_bg_features.shape[2:])
            particle_projection = self.particle_projection(z_v, z_scale_v, z_obj_on_v, z_depth_v, z_features_v,
                                                           z_bg_features_v)
            # [bs * T, n_particles + 1, projection_dim]
            particle_proj_int = particle_projection

            # unroll forward
            particle_proj_int = particle_proj_int.view(bs, -1, *particle_proj_int.shape[1:])
            # [bs, T, n_particles + 1, 2 * projection_dim]
            particle_proj_int = particle_proj_int.permute(0, 2, 1, 3)
            # [bs, n_particles + 1, T, 2 * projection_dim]
            particles_trans = self.particle_transformer(particle_proj_int)
            # [bs * (n_particles + 1), T, projection_dim] or [bs, (n_particles + 1), T, projection_dim]
            particles_trans = particles_trans[:, :, -1]  # [bs, (n_particles + 1), projection_dim]
            # [bs, n_particles + 1, projection_dim]
            # decode transformer output
            # [bs, n_particles + 1, projection_dim]
            particle_decoder_out = self.particle_decoder(particles_trans)
            mu = particle_decoder_out['mu']
            mu = mu.view(bs, 1, *mu.shape[1:])
            logvar = particle_decoder_out['logvar']
            logvar = logvar.view(bs, 1, *logvar.shape[1:])
            obj_on_a = particle_decoder_out['lobj_on_a'].exp().clamp_min(1e-5)
            obj_on_a = obj_on_a.view(bs, 1, *obj_on_a.shape[1:])
            obj_on_b = particle_decoder_out['lobj_on_b'].exp().clamp_min(1e-5)
            obj_on_b = obj_on_b.view(bs, 1, *obj_on_b.shape[1:])
            mu_depth = particle_decoder_out['mu_depth']
            mu_depth = mu_depth.view(bs, 1, *mu_depth.shape[1:])
            logvar_depth = particle_decoder_out['mu_depth']
            logvar_depth = logvar_depth.view(bs, 1, *logvar_depth.shape[1:])
            mu_scale = particle_decoder_out['mu_scale']
            mu_scale = mu_scale.view(bs, 1, *mu_scale.shape[1:])
            logvar_scale = particle_decoder_out['logvar_scale']
            logvar_scale = logvar_scale.view(bs, 1, *logvar_scale.shape[1:])
            mu_features = particle_decoder_out['mu_features']
            mu_features = mu_features.view(bs, 1, *mu_features.shape[1:])
            logvar_features = particle_decoder_out['logvar_features']
            logvar_features = logvar_features.view(bs, 1, *logvar_features.shape[1:])
            mu_bg_features = particle_decoder_out['mu_bg_features']
            mu_bg_features = mu_bg_features.view(bs, 1, *mu_bg_features.shape[1:])
            logvar_bg_features = particle_decoder_out['logvar_bg_features']
            logvar_bg_features = logvar_bg_features.view(bs, 1, *logvar_bg_features.shape[1:])

            if self.predict_delta:
                mu = z[:, -1].unsqueeze(1) + mu
                mu_scale = z_scale[:, -1].unsqueeze(1) + mu_scale
                mu_depth = z_depth[:, -1].unsqueeze(1) + mu_depth
                mu_features = z_features[:, -1].unsqueeze(1) + mu_features
                mu_bg_features = z_bg_features[:, -1].unsqueeze(1) + mu_bg_features

            # if torch.isnan(obj_on_a).any():
            #     print(f'obj_on_a has nan')
            #     torch.nan_to_num_(obj_on_a, nan=0.01)
            # if torch.isnan(obj_on_b).any():
            #     print(f'obj_on_b has nan')
            #     torch.nan_to_num_(obj_on_b, nan=0.01)
            beta_dist = Beta(obj_on_a, obj_on_b)
            if deterministic:
                new_z = mu
                new_z_depth = mu_depth
                new_z_scale = mu_scale
                new_z_features = mu_features
                new_z_bg_features = mu_bg_features
                new_z_obj_on = beta_dist.mean
            else:
                new_z = reparameterize(mu, logvar)
                new_z_depth = reparameterize(mu_depth, logvar_depth)
                new_z_scale = reparameterize(mu_scale, logvar_scale)
                new_z_features = reparameterize(mu_features, logvar_features)
                new_z_bg_features = reparameterize(mu_bg_features, logvar_bg_features)
                new_z_obj_on = beta_dist.sample()

            z = torch.cat([z, new_z], dim=1)
            z_depth = torch.cat([z_depth, new_z_depth], dim=1)
            z_scale = torch.cat([z_scale, new_z_scale], dim=1)
            z_features = torch.cat([z_features, new_z_features], dim=1)
            z_bg_features = torch.cat([z_bg_features, new_z_bg_features], dim=1)
            z_obj_on = torch.cat([z_obj_on, new_z_obj_on.squeeze(-1)], dim=1)

        return z, z_scale, z_obj_on, z_depth, z_features, z_bg_features

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features):
        # forward dynamics
        # z, z_scale: [bs, T, n_particles, 2]
        # z_depth, z_obj_on: [bs, T, n_particles, 1]
        # z_features: [bs, T, n_particles, in_features_dim]
        # z_bg_features: [bs, T, bg_features_dim]
        bs, timestep_horizon, n_particles, _ = z.shape
        # project particles
        z_v = z.reshape(bs * timestep_horizon, *z.shape[2:])
        z_scale_v = z_scale.reshape(bs * timestep_horizon, *z_scale.shape[2:])
        z_obj_on_v = z_obj_on.reshape(bs * timestep_horizon, *z_obj_on.shape[2:])
        z_depth_v = z_depth.reshape(bs * timestep_horizon, *z_depth.shape[2:])
        z_features_v = z_features.reshape(bs * timestep_horizon, *z_features.shape[2:])
        z_bg_features_v = z_bg_features.reshape(bs * timestep_horizon, *z_bg_features.shape[2:])

        particle_projection = self.particle_projection(z_v, z_scale_v, z_obj_on_v, z_depth_v, z_features_v,
                                                       z_bg_features_v)
        # [bs * T, n_particles + 1, projection_dim]
        particle_proj_int = particle_projection

        # unroll forward
        particle_proj_int = particle_proj_int.view(bs, timestep_horizon, *particle_proj_int.shape[1:])
        # [bs, T, n_particles + 1, 2 * projection_dim]
        particle_proj_int = particle_proj_int.permute(0, 2, 1, 3)
        # [bs, n_particles + 1, T, 2 * projection_dim]
        particles_trans = self.particle_transformer(particle_proj_int)
        # [bs, n_particles + 1, T, projection_dim]
        particles_trans = particles_trans.permute(0, 2, 1, 3)
        # [bs, T, n_particles + 1, projection_dim]

        # decode transformer output
        particles_trans = particles_trans.reshape(-1, *particles_trans.shape[2:])
        # [bs * T, n_particles + 1, projection_dim]
        particle_decoder_out = self.particle_decoder(particles_trans)

        mu = particle_decoder_out['mu']
        mu = mu.view(bs, timestep_horizon, *mu.shape[1:])
        logvar = particle_decoder_out['logvar']
        logvar = logvar.view(bs, timestep_horizon, *logvar.shape[1:])
        obj_on_a = particle_decoder_out['lobj_on_a'].exp()
        obj_on_a = obj_on_a.view(bs, timestep_horizon, *obj_on_a.shape[1:])
        obj_on_b = particle_decoder_out['lobj_on_b'].exp()
        obj_on_b = obj_on_b.view(bs, timestep_horizon, *obj_on_b.shape[1:])
        mu_depth = particle_decoder_out['mu_depth']
        mu_depth = mu_depth.view(bs, timestep_horizon, *mu_depth.shape[1:])
        logvar_depth = particle_decoder_out['mu_depth']
        logvar_depth = logvar_depth.view(bs, timestep_horizon, *logvar_depth.shape[1:])
        mu_scale = particle_decoder_out['mu_scale']
        mu_scale = mu_scale.view(bs, timestep_horizon, *mu_scale.shape[1:])
        logvar_scale = particle_decoder_out['logvar_scale']
        logvar_scale = logvar_scale.view(bs, timestep_horizon, *logvar_scale.shape[1:])
        mu_features = particle_decoder_out['mu_features']
        mu_features = mu_features.view(bs, timestep_horizon, *mu_features.shape[1:])
        logvar_features = particle_decoder_out['logvar_features']
        logvar_features = logvar_features.view(bs, timestep_horizon, *logvar_features.shape[1:])
        mu_bg_features = particle_decoder_out['mu_bg_features']
        mu_bg_features = mu_bg_features.view(bs, timestep_horizon, *mu_bg_features.shape[1:])
        logvar_bg_features = particle_decoder_out['logvar_bg_features']
        logvar_bg_features = logvar_bg_features.view(bs, timestep_horizon, *logvar_bg_features.shape[1:])

        if self.predict_delta:
            mu = z + mu
            mu_scale = z_scale + mu_scale
            mu_depth = z_depth + mu_depth
            mu_features = z_features + mu_features
            mu_bg_features = z_bg_features + mu_bg_features

        output_dict = {}

        output_dict['mu'] = mu
        output_dict['logvar'] = logvar

        output_dict['mu_features'] = mu_features
        output_dict['logvar_features'] = logvar_features

        output_dict['obj_on_a'] = obj_on_a.squeeze(-1)
        output_dict['obj_on_b'] = obj_on_b.squeeze(-1)

        output_dict['mu_depth'] = mu_depth
        output_dict['logvar_depth'] = logvar_depth

        output_dict['mu_scale'] = mu_scale
        output_dict['logvar_scale'] = logvar_scale

        output_dict['mu_bg_features'] = mu_bg_features
        output_dict['logvar_bg_features'] = logvar_bg_features

        return output_dict
