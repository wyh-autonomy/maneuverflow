"""TransFuser image and physical LiDAR backbone."""

import math

import timm
import torch
import torch.nn.functional as F
from torch import nn

from .config import ManeuverFlowConfig


class TransfuserBackbone(nn.Module):
    """Multi-scale Fusion Transformer for image + LiDAR feature fusion."""

    def __init__(self, config: ManeuverFlowConfig):

        super().__init__()
        self.config = config

        # Checkpoints restore the full encoder; construction needs no download.
        self.image_encoder = timm.create_model(config.image_architecture, pretrained=False, features_only=True)
        in_channels = int(config.lidar_seq_len)
        if in_channels <= 0:
            raise ValueError(f"LiDAR input channels must be positive, got {in_channels}.")
        self.lidar_input_channels = in_channels

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))

        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
        )
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))
        lidar_time_frames = [1, 1, 1, 1]

        start_index = 0
        # Match feature scales when TIMM exposes an extra stem return layer.
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    config=config,
                    lidar_time_frames=lidar_time_frames[i],
                )
                for i in range(4)
            ]
        )
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )

        self.num_features = self.lidar_encoder.feature_info.info[
            start_index + 3
        ]["num_chs"]

        channel = self.config.current_bev_feature_channels
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=self.config.bev_upsample_factor,
            mode="bilinear",
            align_corners=False,
        )
        self.upsample2 = nn.Upsample(
            size=(
                self.config.lidar_resolution_height
                // self.config.bev_down_sample_factor,
                self.config.lidar_resolution_width
                // self.config.bev_down_sample_factor,
            ),
            mode="bilinear",
            align_corners=False,
        )

        self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

        self.c5_conv = nn.Conv2d(
            self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"],
            channel,
            (1, 1),
        )

    def top_down(self, x):

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))

        return p3

    def forward(self, image, lidar):
        """Fuse image and LiDAR tensors at four encoder scales."""
        if not torch.is_tensor(lidar):
            raise KeyError(
                "ManeuverFlow-TF requires a LiDAR feature tensor."
            )
        image_features, lidar_features = image, lidar

        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        # Some architectures expose an extra stem return layer.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

        for i in range(4):
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

            image_features, lidar_features = self.fuse_features(image_features, lidar_features, i)

        x4 = lidar_features

        # The released topology has no perspective semantic/depth heads.
        image_feature_grid = None

        fused_features = lidar_features

        features = self.top_down(x4)

        return features, fused_features, image_feature_grid

    def forward_layer_block(self, layers, return_layers, features):
        """Advance a TIMM layer iterator to its next return layer."""
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def fuse_features(self, image_features, lidar_features, layer_idx):
        """Add cross-modal Transformer features to both encoder branches."""
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)

        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](image_embd_layer, lidar_embd_layer)
        lidar_features_layer = self.img_channel_to_lidar[layer_idx](lidar_features_layer)

        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features


class GPT(nn.Module):
    """Transformer over concatenated image and LiDAR tokens."""

    def __init__(self, n_embd, config, lidar_time_frames):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1
        self.lidar_seq_len = config.lidar_seq_len
        self.config = config
        self.lidar_time_frames = lidar_time_frames

        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors
                + lidar_time_frames * self.config.lidar_vert_anchors * self.config.lidar_horz_anchors,
                self.n_embd,
            )
        )

        self.drop = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.Sequential(
            *[
                Block(n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop)
                for layer in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor):
        """Fuse pooled image and LiDAR grids shaped (B, C, H, W)."""

        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]

        img_h, img_w = image_tensor.shape[2:4]

        assert self.seq_len == 1
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        image_tensor_out = (
            x[:, : self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors, :]
            .view(bz * self.seq_len, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[
                :,
                self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors :,
                :,
            ]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out


class SelfAttention(nn.Module):
    """Multi-head self-attention with an output projection."""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()

        # Move the head axis ahead of the sequence axis.
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """Pre-normalized Transformer block."""

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
