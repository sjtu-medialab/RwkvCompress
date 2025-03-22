import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torch.utils.cpp_extension import load
from compressai.registry import register_model
from compressai.models import Elic2022Official
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers import (
    CheckerboardMaskedConv2d,
    conv1x1,
    conv3x3,
    sequential_channel_ramp,
)

# T_MAX = 128 * 128  # for training on 256x256 crop
T_MAX = 1024 * 1024  # for inference

current_file_dir = os.path.dirname(os.path.abspath(__file__))
biwkv4_cuda = load(
    name="biwkv4",
    sources=[
        os.path.join(current_file_dir, "cuda/biwkv4_op.cpp"),
        os.path.join(current_file_dir, "cuda/biwkv4_cuda.cu"),
    ],
    verbose=True,
    extra_cuda_cflags=[
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        f"-DTmax={T_MAX}",
    ],
)


class BiWKV4(torch.autograd.Function):
    # Bi-directional WKV version 4,
    # from VisionRWKV https://github.com/OpenGVLab/Vision-RWKV
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device="cuda", memory_format=torch.contiguous_format)
        biwkv4_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device="cuda").contiguous()
        gu = torch.zeros((B, C), device="cuda").contiguous()
        gk = torch.zeros((B, T, C), device="cuda").contiguous()
        gv = torch.zeros((B, T, C), device="cuda").contiguous()
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        biwkv4_cuda.backward(
            B,
            T,
            C,
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            gy.float().contiguous(),
            gw,
            gu,
            gk,
            gv,
        )
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (
                None,
                None,
                None,
                gw.bfloat16(),
                gu.bfloat16(),
                gk.bfloat16(),
                gv.bfloat16(),
            )
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_BiWKV4(B, T, C, w, u, k, v):
    return BiWKV4.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class OmniShift(nn.Module):
    # Reparameterized 5x5 depth-wise convolution,
    # from RestoreRWKV, https://github.com/Yaziwel/Restore-RWKV

    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        out = (
            self.alpha[0] * x
            + self.alpha[1] * out1x1
            + self.alpha[2] * out3x3
            + self.alpha[3] * out5x5
        )
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (
            self.alpha[0] * identity_weight
            + self.alpha[1] * padded_weight_1x1
            + self.alpha[2] * padded_weight_3x3
            + self.alpha[3] * self.conv5x5.weight
        )
        device = self.conv5x5_reparam.weight.device
        combined_weight = combined_weight.to(device)
        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)

    def forward(self, x):
        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif self.training is False and self.repram_flag is True:
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif self.training is False and self.repram_flag is False:
            out = self.conv5x5_reparam(x)

        return out


class SpatialMix_BiV4(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        attn_dim = dim

        self.omni_shift = OmniShift(dim=dim)
        self.key = nn.Linear(dim, attn_dim, bias=False)
        self.value = nn.Linear(dim, attn_dim, bias=False)
        self.receptance = nn.Linear(dim, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, dim, bias=False)

        self.decay = nn.Parameter(torch.randn((self.dim,)))
        self.boost = nn.Parameter(torch.randn((self.dim,)))

    def jit_func(self, x, resolution):
        H, W = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        sr, k, v = self.jit_func(x, resolution)
        x = RUN_BiWKV4(B, T, C, self.decay / T, self.boost / T, k, v)
        x = sr * x
        x = self.output(x)
        return x


class ChannelMix_V4(nn.Module):
    def __init__(self, dim, hidden_rate=4):
        super().__init__()
        self.n_embd = dim
        hidden_dim = int(hidden_rate * dim)

        self.omni_shift = OmniShift(dim=dim)
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, resolution):
        H, W = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv
        return x


class RwkvBlock_BiV4(nn.Module):
    def __init__(self, dim, hidden_rate=4, with_ckpt=False):
        super().__init__()
        self.with_ckpt = with_ckpt

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.att = SpatialMix_BiV4(dim)
        self.ffn = ChannelMix_V4(dim, hidden_rate)
        self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)

    def _forward(self, x):
        B, C, H, W = x.shape
        resolution = (H, W)

        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x

    def forward(self, x):
        if self.with_ckpt and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        else:
            return self._forward(x)


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def form_modules(*modules):
    flattened = []
    for m in modules:
        if isinstance(m, list):
            flattened.extend(m)
        else:
            flattened.append(m)
    return nn.Sequential(*flattened)


class EntropyParametersBlock(nn.Module):
    def __init__(self, dim, out_dim, expansion_factor=2, **kwargs):
        super().__init__()
        
        hidden_dim = int(expansion_factor * out_dim)
        self.mix = nn.Conv2d(dim, out_dim, 1)
        self.norm = nn.LayerNorm(out_dim)
        self.key = nn.Linear(out_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, out_dim, bias=False)
        self.receptance = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.mix(x)
        identity = x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x + identity


@register_model("LALIC")
class LALIC(Elic2022Official):
    def __init__(
        self,
        N=128,
        M=320,
        dims=[96, 144, 256, 320, 256, 192],
        depths=[2, 4, 6, 6],
        groups=None,
        use_ckpt=False,
        **kwargs,
    ):
        super().__init__(N=N, M=M, groups=groups, **kwargs)
        # self.N = N
        # self.M = M
        N1, N2, N3, N4, N5, N6 = dims
        L1, L2, L3, L4 = depths
        M = N4

        # flatten the list
        self.g_a = form_modules(
            conv(3, N1, kernel_size=5),
            [RwkvBlock_BiV4(N1) for _ in range(L1)],
            conv(N1, N2, kernel_size=3),
            [RwkvBlock_BiV4(N2) for i in range(L2)],
            conv(N2, N3, kernel_size=3),
            [RwkvBlock_BiV4(N3) for _ in range(L3)],
            conv(N3, N4, kernel_size=3),
        )

        self.g_s = form_modules(
            deconv(N4, N3, kernel_size=3),
            [RwkvBlock_BiV4(N3) for _ in range(L3)],
            deconv(N3, N2, kernel_size=3),
            [RwkvBlock_BiV4(N2) for _ in range(L2)],
            deconv(N2, N1, kernel_size=3),
            [RwkvBlock_BiV4(N1) for _ in range(L1)],
            deconv(N1, 3, kernel_size=5),
        )

        self.h_a = form_modules(
            conv(N4, N5, kernel_size=5),
            [RwkvBlock_BiV4(N5) for _ in range(L4)],
            conv(N5, N6, kernel_size=5),
        )

        self.h_s = form_modules(
            deconv(N6, N5, kernel_size=5),
            [RwkvBlock_BiV4(N5) for _ in range(L4)],
            deconv(N5, N4, kernel_size=5),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": nn.Sequential(
                conv3x3(sum(self.groups[:k]), M),
                RwkvBlock_BiV4(M, hidden_rate=8),
                RwkvBlock_BiV4(M, hidden_rate=8),
                conv1x1(M, self.groups[k] * 2),
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + M,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=EntropyParametersBlock,
                make_act=nn.Identity,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N6),
                    h_a=self.h_a,
                    h_s=self.h_s,
                    quantizer="ste",
                ),
            },
        )

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net
