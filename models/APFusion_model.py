import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange


class APFusion(nn.Module):
    def __init__(self, model_clip, inp_A_channels=1, inp_B_channels=1, out_channels=1,
                 dim=48, num_blocks=[1, 1, 1, 1],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(APFusion, self).__init__()

        self.model_clip = model_clip
        self.model_clip.eval()

        self.encoder_A = Encoder_A(inp_channels=inp_A_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.encoder_B = Encoder_B(inp_channels=inp_B_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.attention_spatial = Attention_spatial(dim * 2 ** 3)
        self.CFM = CFM(dim * 2 ** 3, dim)

        self.feature_fusion_4 = TextGuidedDWTFusion(channels=dim * 2 ** 3)
        self.prompt_guidance_4 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 3)

        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.reduce_chan_level4 = nn.Conv2d(int(dim * 2 ** 4), int(dim * 2 ** 3), kernel_size=1, bias=bias)

        self.feature_fusion_3 = TextGuidedDWTFusion(channels=dim * 2 ** 2)
        self.prompt_guidance_3 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 2)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.feature_fusion_2 = TextGuidedDWTFusion(channels=dim * 2 ** 1)
        self.prompt_guidance_2 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 1)
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.feature_fusion_1 = TextGuidedDWTFusion(channels=dim)
        self.prompt_guidance_1 = FeatureWiseAffine(in_channels=512, out_channels=dim)
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 0), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 0), kernel_size=1, bias=bias)
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 0), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 0), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img_A, inp_img_B, text_vi, text_ir, text_fuse):
        b = inp_img_A.shape[0]
        text_vi_features = self.get_text_feature(text_vi.expand(b, -1)).to(inp_img_A.dtype)
        text_ir_features = self.get_text_feature(text_ir.expand(b, -1)).to(inp_img_A.dtype)
        text_fuse_features = self.get_text_feature(text_fuse.expand(b, -1)).to(inp_img_A.dtype)

        out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A = self.encoder_A(inp_img_A,
                                                                                                text_vi_features)
        out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B = self.encoder_B(inp_img_B,
                                                                                                text_ir_features)

        out_enc_level4 = self.CFM(out_enc_level4_A, out_enc_level4_B)

        out_enc_level4, _, _ = self.attention_spatial(out_enc_level4)

        inp_dec_level4 = out_enc_level4

        out_dec_level4 = self.decoder_level4(inp_dec_level4)
        out_dec_level4 = self.prompt_guidance_4(out_dec_level4, text_fuse_features)
        out_enc_level4 = self.feature_fusion_4(out_enc_level4_A, out_enc_level4_B, text_fuse_features)
        out_dec_level4 = torch.cat([out_dec_level4, out_enc_level4], 1)
        out_dec_level4 = self.reduce_chan_level4(out_dec_level4)

        inp_dec_level3 = self.up4_3(out_dec_level4)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        out_dec_level3 = self.prompt_guidance_3(out_dec_level3, text_fuse_features)
        out_enc_level3 = self.feature_fusion_3(out_enc_level3_A, out_enc_level3_B, text_fuse_features)
        out_dec_level3 = torch.cat([out_dec_level3, out_enc_level3], 1)
        out_dec_level3 = self.reduce_chan_level3(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.prompt_guidance_2(out_dec_level2, text_fuse_features)
        out_enc_level2 = self.feature_fusion_2(out_enc_level2_A, out_enc_level2_B, text_fuse_features)
        out_dec_level2 = torch.cat([out_dec_level2, out_enc_level2], 1)
        out_dec_level2 = self.reduce_chan_level2(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.prompt_guidance_1(out_dec_level1, text_fuse_features)
        out_enc_level1 = self.feature_fusion_1(out_enc_level1_A, out_enc_level1_B, text_fuse_features)
        out_dec_level1 = torch.cat([out_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.reduce_chan_level1(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature


import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT(nn.Module):
    """
    Haar DWT for feature maps
    Input:  [B, C, H, W]
    Output: LL, LH, HL, HH ∈ [B, C, H/2, W/2]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # even / odd sampling
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) * 0.5
        LH = (x00 - x01 + x10 - x11) * 0.5
        HL = (x00 + x01 - x10 - x11) * 0.5
        HH = (x00 - x01 - x10 + x11) * 0.5

        return LL, LH, HL, HH


class IDWT(nn.Module):
    """
    Inverse Haar DWT
    Input: LL, LH, HL, HH ∈ [B, C, H, W]
    Output: [B, C, 2H, 2W]
    """

    def __init__(self):
        super().__init__()

    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=LL.device)

        out[:, :, 0::2, 0::2] = (LL + LH + HL + HH) * 0.5
        out[:, :, 0::2, 1::2] = (LL - LH + HL - HH) * 0.5
        out[:, :, 1::2, 0::2] = (LL + LH - HL - HH) * 0.5
        out[:, :, 1::2, 1::2] = (LL - LH - HL + HH) * 0.5

        return out


class FrequencyExpert(nn.Module):
    """
    Frequency expert.

    Input:
        Cat(F_ir^b, F_vi^b): [B, 2C, H, W]

    Output:
        F_fused^b: [B, C, H, W]
    """

    def __init__(self, channels, hidden_channels=None):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channels, channels, kernel_size=1)
        )

    def forward(self, ir_band, vi_band):
        x = torch.cat([ir_band, vi_band], dim=1)
        fused_band = self.net(x)
        return fused_band


class TextGuidedDWTFusion(nn.Module):
    def __init__(self, channels, text_dim=512):
        super().__init__()

        self.dwt = DWT()
        self.idwt = IDWT()

        self.low_freq_expert = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

        self.high_freq_expert = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, F_ir, F_vi, text_embed):
        # --- DWT ---
        ir_LL, ir_LH, ir_HL, ir_HH = self.dwt(F_ir)
        vi_LL, vi_LH, vi_HL, vi_HH = self.dwt(F_vi)

        # --- low-frequency expert for LL ---
        LL = self.low_freq_expert(
            torch.cat([ir_LL, vi_LL], dim=1)
        )

        # --- high-frequency expert for LH / HL / HH ---
        LH = self.high_freq_expert(
            torch.cat([ir_LH, vi_LH], dim=1)
        )

        HL = self.high_freq_expert(
            torch.cat([ir_HL, vi_HL], dim=1)
        )

        HH = self.high_freq_expert(
            torch.cat([ir_HH, vi_HH], dim=1)
        )

        # --- IDWT ---
        F_fused = self.idwt(LL, LH, HL, HH)

        return F_fused


class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B


class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input, key, value


class Cross_attention1(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B, k, v):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        key_B = k
        value_B = v

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_B


class CFM(nn.Module):
    def __init__(self, in_channels, dim, n_head=1, norm_groups=16):
        super().__init__()
        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.cross_attention1 = Cross_attention1(dim * 2 ** 3)
        self.x_attention_spatial = Attention_spatial(dim * 2 ** 3)
        self.reduce_x = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.reduce_f = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, vi, ir):
        vi_1, ir_1 = self.cross_attention(vi, ir)
        x = self.reduce_x(torch.cat([ir, vi], dim=1))
        f = self.reduce_f(torch.cat([ir_1, vi_1], dim=1))
        x, k_x, v_x = self.x_attention_spatial(x)
        f = self.cross_attention1(x, f, k_x, v_x)
        return f


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        # MLP 输出 gamma 和 beta
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        B, C, H, W = x.shape
        # 生成 gamma 和 beta

        gamma, beta = self.MLP(text_embed).view(B, -1, 1, 1).chunk(2, dim=1)  # [B, C, 1, 1]

        # 门控调制
        x = (1 + gamma) * x + beta

        return x


class FeatureMap_guide(nn.Module):

    def __init__(self, kernel_size=7, negative_slope=0.2):
        super(FeatureMap_guide, self).__init__()

        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.avg_branch = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        )

        self.max_branch = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W], corresponding to \bar{F}_m^i

        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        M_avg = self.avg_branch(avg_out)  # [B, 1, H, W]
        M_max = self.max_branch(max_out)  # [B, 1, H, W]

        gamma = self.sigmoid(M_avg + M_max)  # [B, 1, H, W]

        out = x * gamma + x  # residual modulation

        return out


class Encoder_A(nn.Module):
    def __init__(self, inp_channels=3, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_A, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Conv2d(int(dim), int(dim), kernel_size=3, stride=1, padding=1)

        self.map_guide_level1 = FeatureMap_guide(kernel_size=7)
        self.prompt_guidance_1 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 0)

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.encoder_level2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=3, stride=1, padding=1)

        self.map_guide_level2 = FeatureMap_guide(kernel_size=7)
        self.prompt_guidance_2 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 1)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.encoder_level3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=3, stride=1, padding=1)

        self.map_guide_level3 = FeatureMap_guide(kernel_size=7)
        self.prompt_guidance_3 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 2)

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        self.encoder_level4 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 3), kernel_size=3, stride=1, padding=1)

        self.map_guide_level4 = FeatureMap_guide(kernel_size=7)
        self.prompt_guidance_4 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 3)

    def forward(self, inp_img_A, text_vi):
        inp_enc_level1_A = self.patch_embed(inp_img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A)
        out_enc_level1_A = self.prompt_guidance_1(out_enc_level1_A, text_vi)
        out_enc_level1_A = self.map_guide_level1(out_enc_level1_A)

        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A)
        out_enc_level2_A = self.prompt_guidance_2(out_enc_level2_A, text_vi)
        out_enc_level2_A = self.map_guide_level2(out_enc_level2_A)

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A)
        out_enc_level3_A = self.prompt_guidance_3(out_enc_level3_A, text_vi)
        out_enc_level3_A = self.map_guide_level3(out_enc_level3_A)

        inp_enc_level4_A = self.down3_4(out_enc_level3_A)
        out_enc_level4_A = self.encoder_level4(inp_enc_level4_A)
        out_enc_level4_A = self.prompt_guidance_4(out_enc_level4_A, text_vi)
        out_enc_level4_A = self.map_guide_level4(out_enc_level4_A)

        return out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A


class Encoder_B(nn.Module):
    def __init__(self, inp_channels=1, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_B, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Conv2d(int(dim), int(dim), kernel_size=3, stride=1, padding=1)

        self.map_guide_level1 = FeatureMap_guide(kernel_size=7)
        self.prompt_guidance_1 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 0)

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.encoder_level2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=3, stride=1, padding=1)

        self.map_guide_level2 = FeatureMap_guide(kernel_size=7)
        self.prompt_guidance_2 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 1)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.encoder_level3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=3, stride=1, padding=1)

        self.map_guide_level3 = FeatureMap_guide(kernel_size=7)
        self.prompt_guidance_3 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 2)

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        self.encoder_level4 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 3), kernel_size=3, stride=1, padding=1)

        self.map_guide_level4 = FeatureMap_guide(kernel_size=7)
        self.prompt_guidance_4 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 3)

    def forward(self, inp_img_B, text_ir):
        inp_enc_level1_B = self.patch_embed(inp_img_B)
        out_enc_level1_B = self.encoder_level1(inp_enc_level1_B)
        out_enc_level1_B = self.prompt_guidance_1(out_enc_level1_B, text_ir)
        out_enc_level1_B = self.map_guide_level1(out_enc_level1_B)

        inp_enc_level2_B = self.down1_2(out_enc_level1_B)
        out_enc_level2_B = self.encoder_level2(inp_enc_level2_B)
        out_enc_level2_B = self.prompt_guidance_2(out_enc_level2_B, text_ir)
        out_enc_level2_B = self.map_guide_level2(out_enc_level2_B)

        inp_enc_level3_B = self.down2_3(out_enc_level2_B)
        out_enc_level3_B = self.encoder_level3(inp_enc_level3_B)
        out_enc_level3_B = self.prompt_guidance_3(out_enc_level3_B, text_ir)
        out_enc_level3_B = self.map_guide_level3(out_enc_level3_B)

        inp_enc_level4_B = self.down3_4(out_enc_level3_B)
        out_enc_level4_B = self.encoder_level4(inp_enc_level4_B)
        out_enc_level4_B = self.prompt_guidance_4(out_enc_level4_B, text_ir)
        out_enc_level4_B = self.map_guide_level4(out_enc_level4_B)

        return out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B


class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))  # 可选，增加非线性，效果更平滑)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)