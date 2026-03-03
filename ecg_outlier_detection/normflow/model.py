# Copyright (c) 2020 Vincent Stimper
# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import math

import normflows as nf
import torch
import torch.nn as nn


# The following class "ConvNet1d" was adapted from normalizing-flows
# Source: https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/nets/cnn.py
# Licensed under the MIT License: https://opensource.org/license/mit
class ConvNet1d(nn.Module):
    """
    Convolutional Neural Network with leaky ReLU nonlinearities
    """

    def __init__(
        self,
        channels,
        kernel_size,
        leaky=0.0,
        init_zeros=True,
        actnorm=False,
        weight_std=None,
    ):
        """Constructor

        Args:
          channels: List of channels of conv layers, first entry is in_channels
          kernel_size: List of kernel sizes, same for height and width
          leaky: Leaky part of ReLU
          init_zeros: Flag whether last layer shall be initialized with zeros
          scale_output: Flag whether to scale output with a log scale parameter
          logscale_factor: Constant factor to be multiplied to log scaling
          actnorm: Flag whether activation normalization shall be done after each conv layer except output
          weight_std: Fixed std used to initialize every layer
        """
        super().__init__()
        # Build network
        net = nn.ModuleList([])
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv1d(
                channels[i],
                channels[i + 1],
                kernel_size[i],
                padding=kernel_size[i] // 2,
                bias=(not actnorm),
            )
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(nf.utils.nn.ActNorm((channels[i + 1], 1)))
            net.append(nn.LeakyReLU(leaky))
        i = len(kernel_size)
        net.append(
            nn.Conv1d(
                channels[i - 1],
                channels[i],
                kernel_size[i - 1],
                padding=kernel_size[i - 1] // 2,
            )
        )
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


# The following class "Invertible1x1Conv1dFlexible" was adapted from normalizing-flows
# Source: https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/mixing.py
# Licensed under the MIT License: https://opensource.org/license/mit
class Invertible1x1Conv1dFlexible(nf.flows.base.Flow):
    """
    Invertible 1x1 convolution introduced in the Glow paper
    Assumes 3d input/output tensors of the form NCW
    """

    def __init__(self, num_features, use_lu=False, mode="channel"):
        """Constructor

        Args:
          num_features: Number of channels of the data
          use_lu: Flag whether to parametrize weights through the LU decomposition
        """
        super().__init__()
        self.mode = mode
        self.num_features = num_features
        self.use_lu = use_lu
        Q, _ = torch.linalg.qr(torch.randn(self.num_features, self.num_features))
        if use_lu:
            P, L, U = torch.lu_unpack(*Q.lu())
            self.register_buffer("P", P)  # remains fixed during optimization
            self.L = nn.Parameter(L)  # lower triangular portion
            S = U.diag()  # "crop out" the diagonal to its own parameter
            self.register_buffer("sign_S", torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(
                torch.triu(U, diagonal=1)
            )  # "crop out" diagonal, stored in S
            self.register_buffer("eye", torch.diag(torch.ones(self.num_features)))
        else:
            self.W = nn.Parameter(Q)

    def _assemble_W(self, inverse=False):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(
            self.sign_S * torch.exp(self.log_S)
        )
        if inverse:
            if self.log_S.dtype == torch.float64:
                L_inv = torch.inverse(L)
                U_inv = torch.inverse(U)
            else:
                L_inv = torch.inverse(L.double()).type(self.log_S.dtype)
                U_inv = torch.inverse(U.double()).type(self.log_S.dtype)
            W = U_inv @ L_inv @ self.P.t()
        else:
            W = self.P @ L @ U
        return W

    def forward(self, z):
        if self.use_lu:
            W = self._assemble_W(inverse=True)
            log_det = -torch.sum(self.log_S)
        else:
            W_dtype = self.W.dtype
            if W_dtype == torch.float64:
                W = torch.inverse(self.W)
            else:
                W = torch.inverse(self.W.double()).type(W_dtype)
            W = W.view(*W.size(), 1, 1)
            log_det = -torch.slogdet(self.W)[1]
        W = W.view(self.num_features, self.num_features, 1)
        if self.mode == "channel":
            z_ = torch.nn.functional.conv1d(z, W)
        elif self.mode == "time":
            z = z.permute(0, 2, 1)
            z_ = torch.nn.functional.conv1d(z, W)
            z_ = z_.permute(0, 2, 1)
        log_det = log_det * z.size(2)
        return z_, log_det

    def inverse(self, z):
        if self.use_lu:
            W = self._assemble_W()
            log_det = torch.sum(self.log_S)
        else:
            W = self.W
            log_det = torch.slogdet(self.W)[1]
        W = W.view(self.num_features, self.num_features, 1)
        if self.mode == "channel":
            z_ = torch.nn.functional.conv1d(z, W)
        elif self.mode == "time":
            z = z.permute(0, 2, 1)
            z_ = torch.nn.functional.conv1d(z, W)
            z_ = z_.permute(0, 2, 1)
        log_det = log_det * z.size(2)
        return z_, log_det


# The following class "Squeeze1d" was adapted from normalizing-flows
# Source: https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/reshape.py
# Licensed under the MIT License: https://opensource.org/license/mit
class Squeeze1d(nf.flows.base.Flow):
    """
    Squeeze operation of multi-scale architecture, RealNVP or Glow paper
    """

    def __init__(self, ratio=2):
        """
        Constructor
        """
        super().__init__()
        self.ratio = ratio

    def forward(self, z):
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1] // self.ratio, self.ratio, s[2])
        z = z.permute(0, 1, 3, 2).contiguous()
        z = z.view(s[0], s[1] // self.ratio, self.ratio * s[2])
        return z, log_det

    def inverse(self, z):
        log_det = 0
        s = z.size()
        z = z.view(*s[:2], s[2] // self.ratio, self.ratio)
        z = z.permute(0, 1, 3, 2).contiguous()
        z = z.view(s[0], self.ratio * s[1], s[2] // self.ratio)
        return z, log_det


# The following class "SplitFlexible" was adapted from normalizing-flows
# Source: https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/reshape.py
# Licensed under the MIT License: https://opensource.org/license/mit
class SplitFlexible(nf.flows.base.Flow):
    """
    Split features into two sets
    """

    def __init__(self, mode="channel", ratio=0.5):
        """Constructor

        The splitting mode can be:

        - channel: Splits first feature dimension, usually channels, into two halfs
        - channel_inv: Same as channel, but with z1 and z2 flipped
        - checkerboard: Splits features using a checkerboard pattern (last feature dimension must be even)
        - checkerboard_inv: Same as checkerboard, but with inverted coloring

        Args:
         mode: splitting mode
        """
        super().__init__()
        self.mode = mode
        self.ratio = ratio

    def forward(self, z):
        if self.mode == "channel":
            z1 = z[:, 0 : math.ceil(z.shape[1] * (1 - self.ratio))]
            z2 = z[:, math.ceil(z.shape[1] * (1 - self.ratio)) :]
        elif self.mode == "channel_inv":
            z2 = z[:, 0 : math.ceil(z.shape[1] * (1 - self.ratio))]
            z1 = z[:, math.ceil(z.shape[1] * (1 - self.ratio)) :]
        elif self.mode == "time":
            z1 = z[:, :, 0 : math.ceil(z.shape[2] * (1 - self.ratio))]
            z2 = z[:, :, math.ceil(z.shape[2] * (1 - self.ratio)) :]
        elif "checkerboard" in self.mode:
            n_dims = z.dim()
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
            cb = cb1 if "inv" in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
            cb = cb.to(z.device)
            z_size = z.size()
            z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(
                *z_size[:-1], -1
            )
            z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(
                *z_size[:-1], -1
            )
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        if self.mode == "channel":
            z = torch.cat([z1, z2], 1)
        elif self.mode == "channel_inv":
            z = torch.cat([z2, z1], 1)
        elif self.mode == "time":
            z = torch.cat([z1, z2], 2)
        elif "checkerboard" in self.mode:
            n_dims = z1.dim()
            z_size = list(z1.size())
            z_size[-1] *= 2
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
            cb = cb1 if "inv" in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
            cb = cb.to(z1.device)
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det


# The following class "MergeFlexible" was adapted from normalizing-flows
# Source: https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/reshape.py
# Licensed under the MIT License: https://opensource.org/license/mit
class MergeFlexible(SplitFlexible):
    """
    Same as Split but with forward and backward pass interchanged
    """

    def __init__(self, mode="channel", ratio=0.5):
        super().__init__(mode, ratio)

    def forward(self, z):
        return super().inverse(z)

    def inverse(self, z):
        return super().forward(z)


# The following class "AffineCouplingFlexible" was adapted from normalizing-flows
# Source: https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/affine/coupling.py
# Licensed under the MIT License: https://opensource.org/license/mit
class AffineCouplingFlexible(nf.flows.base.Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map="exp", param_dim="channel"):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid scale when sampling from the model
        """
        super().__init__()
        self.add_module("param_map", param_map)
        self.scale = scale
        self.scale_map = scale_map
        self.param_dim = param_dim

    def forward(self, z):
        """
        z is a list of z1 and z2; ```z = [z1, z2]```
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1

        Args:
          z
        """
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            if self.param_dim == "channel":
                shift = param[:, 0::2, ...]
                scale_ = param[:, 1::2, ...]
            elif self.param_dim == "time":
                shift = param[:, :, 0::2, ...]
                scale_ = param[:, :, 1::2, ...]
            if self.scale_map == "exp":
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 + param
            log_det = nf.flows.base.zero_log_det_like_z(z2)
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            if self.param_dim == "channel":
                shift = param[:, 0::2, ...]
                scale_ = param[:, 1::2, ...]
            elif self.param_dim == "time":
                shift = param[:, :, 0::2, ...]
                scale_ = param[:, :, 1::2, ...]
            if self.scale_map == "exp":
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 - param
            log_det = nf.flows.base.zero_log_det_like_z(z2)
        return [z1, z2], log_det


# The following class "AffineCouplingBlockFlexible" was adapted from normalizing-flows
# Source: https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/affine/coupling.py
# Licensed under the MIT License: https://opensource.org/license/mit
class AffineCouplingBlockFlexible(nf.flows.base.Flow):
    """
    Affine Coupling layer including split and merge operation
    """

    def __init__(
        self,
        param_map,
        scale=True,
        scale_map="exp",
        split_mode="channel",
        param_dim="channel",
    ):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow
          split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [SplitFlexible(split_mode)]
        # Affine coupling layer
        self.flows += [
            AffineCouplingFlexible(param_map, scale, scale_map, param_dim=param_dim)
        ]
        # Merge layer
        self.flows += [MergeFlexible(split_mode)]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot


# The following class "GlowBlock1D" was adapted from normalizing-flows
# Source: https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/affine/glow.py
# Licensed under the MIT License: https://opensource.org/license/mit
class GlowBlock1D(nf.flows.base.Flow):
    """Glow: Generative Flow with Invertible 1×1 Convolutions, [arXiv: 1807.03039](https://arxiv.org/abs/1807.03039)

    One Block of the Glow model, comprised of

    - MaskedAffineFlow (affine coupling layer)
    - Invertible1x1Conv (dropped if there is only one channel)
    - ActNorm (first batch used for initialization)
    """

    def __init__(
        self,
        channels,
        timesteps,
        filter_list,
        kernel_list,
        scale=True,
        scale_map="sigmoid",
        split_mode="channel",
        permute_mode="channel",
        leaky=0.0,
        init_zeros=True,
        use_lu=True,
        net_actnorm=False,
        network="CNN",
        param_dim="channel",
    ):
        """Constructor

        Args:
          channels: Number of channels of the data
          hidden_channels: number of channels in the hidden layer of the ConvNet
          scale: Flag, whether to include scale in affine coupling layer
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow
          split_mode: Splitting mode, for possible values see Split class
          leaky: Leaky parameter of LeakyReLUs of ConvNet2d
          init_zeros: Flag whether to initialize last conv layer with zeros
          use_lu: Flag whether to parametrize weights through the LU decomposition in invertible 1x1 convolution layers
          logscale_factor: Factor which can be used to control the scale of the log scale factor, see [source](https://github.com/openai/glow)
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Coupling layer
        num_param = 2 if scale else 1
        if network == "CNN":
            if "channel" == split_mode:
                channels_ = ((channels + 1) // 2,) + filter_list
                channels_ += (num_param * (channels // 2),)
            elif "channel_inv" == split_mode:
                channels_ = (channels // 2,) + filter_list
                channels_ += (num_param * ((channels + 1) // 2),)
            elif "checkerboard" in split_mode:
                channels_ = (channels,) + filter_list
                channels_ += (num_param * channels,)
            else:
                raise NotImplementedError("Mode " + split_mode + " is not implemented.")
            param_map = ConvNet1d(
                channels_, kernel_list, leaky, init_zeros, actnorm=net_actnorm
            )

        elif network == "MLP":
            channels_ = (timesteps // 2,) + filter_list
            channels_ += (num_param * (timesteps // 2),)
            param_map = nf.nets.mlp.MLP(channels_, leaky, init_zeros=init_zeros)
        self.flows += [
            AffineCouplingBlockFlexible(
                param_map, scale, scale_map, split_mode, param_dim
            )
        ]
        # Invertible 1x1 convolution
        if permute_mode == "channel":
            self.flows += [
                Invertible1x1Conv1dFlexible(channels, use_lu, mode=permute_mode)
            ]
            self.flows += [nf.flows.normalization.ActNorm((channels,) + (1,))]
        elif permute_mode == "time":
            self.flows += [
                Invertible1x1Conv1dFlexible(timesteps, use_lu, mode=permute_mode)
            ]
            self.flows += [nf.flows.normalization.ActNorm((1, timesteps))]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot


def get_model_size(model):
    param_size = 0
    param_n = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_n += param.nelement()

    return param_n


def get_multiscale_model(
    filter_list,
    kernel_list,
    num_blocks,
    input_shape,
    num_levels,
    split_ratio_list,
    squeeze_ratio_list,
    split_mode,
    permute_mode,
    latent_split_mode,
    network,
    param_dim,
):
    q0 = []
    merges = []
    flows = []
    current_shape = input_shape
    for i in range(num_levels):
        flows_ = []
        flows_ += [Squeeze1d(ratio=squeeze_ratio_list[i])]
        current_shape = (
            int(current_shape[0] * squeeze_ratio_list[i]),
            int(current_shape[1] // squeeze_ratio_list[i]),
        )
        for j in range(num_blocks[i]):
            if network == "CNN":
                flows_.append(
                    GlowBlock1D(
                        channels=current_shape[0],
                        timesteps=current_shape[1],
                        filter_list=tuple(filter_list),
                        kernel_list=kernel_list,
                        scale_map="exp",
                        split_mode=split_mode,
                        permute_mode=permute_mode,
                        network=network,
                        param_dim=param_dim,
                    )
                )
            elif network == "MLP":
                flows_.append(
                    GlowBlock1D(
                        channels=current_shape[0],
                        timesteps=current_shape[1],
                        filter_list=tuple(filter_list),
                        kernel_list=None,
                        scale_map="exp",
                        split_mode=split_mode,
                        permute_mode=permute_mode,
                        network=network,
                        param_dim=param_dim,
                    )
                )
        flows += [reversed(flows_)]

        if i < (num_levels - 1):
            merges += [MergeFlexible(ratio=split_ratio_list[i], mode=latent_split_mode)]
            if latent_split_mode == "channel":
                latent_shape = (
                    int(current_shape[0] * split_ratio_list[i]),
                    current_shape[1],
                )
                current_shape = (
                    math.ceil(current_shape[0] * (1 - split_ratio_list[i])),
                    current_shape[1],
                )
            elif latent_split_mode == "time":
                latent_shape = (
                    current_shape[0],
                    int(current_shape[1] * split_ratio_list[i]),
                )
                current_shape = (
                    current_shape[0],
                    math.ceil(current_shape[1] * (1 - split_ratio_list[i])),
                )
            q0 += [nf.distributions.DiagGaussian(latent_shape)]
        else:
            q0 += [nf.distributions.DiagGaussian(current_shape)]

    model = nf.MultiscaleFlow(
        reversed(q0), reversed(flows), reversed(merges), class_cond=False
    )

    return model
